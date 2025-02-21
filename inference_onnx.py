import time
from scipy.special import softmax
from collections import namedtuple
from typing import Any, Optional

import torch
from tqdm import tqdm
from litgpt import Tokenizer
import onnx
import onnxruntime
import glob
import os
from utils.snac_utils import layershift, reconscruct_snac, reconstruct_tensors
import soundfile as sf
import whisper
import numpy as np

text_vocabsize = 151936
text_specialtokens = 64
audio_vocabsize = 4096
audio_specialtokens = 64

padded_text_vocabsize = text_vocabsize + text_specialtokens
padded_audio_vocabsize = audio_vocabsize + audio_specialtokens

_eot = text_vocabsize
_pad_t = text_vocabsize + 1
_input_t = text_vocabsize + 2
_answer_t = text_vocabsize + 3
_asr = text_vocabsize + 4

_eoa = audio_vocabsize
_pad_a = audio_vocabsize + 1
_input_a = audio_vocabsize + 2
_answer_a = audio_vocabsize + 3
_split = audio_vocabsize + 4
_image = audio_vocabsize + 5
_eoimage = audio_vocabsize + 6


def load_audio(path):
  audio = whisper.load_audio(path)
  duration_ms = (len(audio) / 16000) * 1000
  audio = whisper.pad_or_trim(audio)
  mel = whisper.log_mel_spectrogram(audio)
  return mel.numpy(), int(duration_ms / 20) + 1


def get_input_ids_whisper(
    mel: np.ndarray, leng: int, step: int, whispermodel: onnxruntime.InferenceSession,
    special_token_a=_answer_a, special_token_t=_answer_t,
):

  mel = mel[np.newaxis, ...]
  # mel_golden = np.load(f"output/compare/adapter/mel_{step}.npy")
  # assert np.allclose(mel, mel_golden)
  # whisper_model = whisper.load_model('/Users/lisa/Documents/mini-omni2-k230/checkpoint/small.pt').to(device='cpu:0')
  # with torch.no_grad():
  #   audio_feature = whisper_model.embed_audio(torch.tensor(mel)).numpy()[:,:leng,:]
  (audio_feature,) = whispermodel.run(None, {'mel': mel})
  audio_feature: np.ndarray = audio_feature[:, :leng, :]
  # audio_feature_golden = np.load(f"output/compare/adapter/audio_feature_{step}.npy")
  # assert np.allclose(audio_feature, audio_feature_golden,atol=1e-3)

  input_ids = []
  for i in range(7):
    input_ids_item = []
    input_ids_item.append(layershift(_input_a, i))
    input_ids_item += [layershift(_pad_a, i)] * leng
    input_ids_item += [(layershift(_eoa, i)), layershift(special_token_a, i)]
    input_ids.append(np.array(input_ids_item)[np.newaxis, ...])
  input_id_T = np.array([_input_t] + [_pad_t] * leng + [_eot, special_token_t])
  input_ids.append(input_id_T[np.newaxis, ...])
  return audio_feature, input_ids


def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
  if torch._dynamo.is_compiling():
    # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
    distribution = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / distribution, dim=-1, keepdim=True)
  return torch.multinomial(probs, num_samples=1)


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
  sorted_logits, sorted_indices = torch.sort(logits, descending=False)
  cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
  # Example:
  # sorted_probs=[0.1, 0.15, 0.2, 0.25, 0.3] -> sorted_cumprobs=[0.1, 0.25, 0.45, 0.7, 1.0]
  # sorted_indices_to_remove = [1, 1, 0, 0, 0] if top_p=0.7
  sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
  # Keep at least 1 token always to prevent the case where no token is selected
  # In this case the most probable one is always kept
  sorted_indices_to_remove[-1:] = 0
  indices_to_remove = sorted_indices_to_remove.scatter(
      0, sorted_indices, sorted_indices_to_remove
  )
  logits = logits.masked_fill(indices_to_remove, float("-inf"))
  return logits


def sample(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> np.ndarray:
  logits = torch.tensor(logits)
  if top_p < 0.0 or top_p > 1.0:
    raise ValueError(f"top_p must be in [0, 1], got {top_p}")
  logits = logits[0, -1]
  # optionally crop the logits to only the top k options
  if top_k is not None:
    v, i = torch.topk(logits, min(top_k, logits.size(-1)))
    # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
    logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
  # optionally scale the logits and sample from a probability distribution
  if temperature > 0.0 or top_p > 0.0:
    if temperature > 0.0:
      logits = logits / temperature
    # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
    if top_p < 1.0:
      logits = sample_top_p(logits, top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return multinomial_num_samples_1(probs).numpy()
  return torch.argmax(logits, dim=-1, keepdim=True).numpy()


def next_token_A1T2(
    lit_gpt: onnxruntime.InferenceSession,
    input_embs: np.ndarray,
    input_pos: np.ndarray,
    past_ks: np.ndarray,
    past_vs: np.ndarray,
    step: int,
    sub_step: int,
    **kwargs: Any,
) -> np.ndarray:

  # input_embs_golden = np.load(f"output/compare/lit_gpt/{step}_{sub_step}_input_embs.npy")
  # past_ks_golden = np.load(f"output/compare/lit_gpt/{step}_{sub_step}_past_ks.npy")
  # past_vs_golden = np.load(f"output/compare/lit_gpt/{step}_{sub_step}_past_vs.npy")
  # input_pos_golden = np.load(f"output/compare/lit_gpt/{step}_{sub_step}_input_pos.npy")
  # assert np.allclose(input_embs, input_embs_golden)
  # assert np.allclose(past_ks, past_ks_golden)
  # assert np.allclose(past_vs, past_vs_golden)
  # assert np.allclose(input_pos, input_pos_golden)
  (logits_a, logit_t, next_ks, next_vs) = lit_gpt.run(None, {
      "input_embs": input_embs, "past_keys": past_ks, "past_values": past_vs, "input_pos": input_pos})
  # logits_a_golden = np.save(f"output/compare/lit_gpt/{step}_{sub_step}_logits_a.npy")
  # logit_t_golden = np.save(f"output/compare/lit_gpt/{step}_{sub_step}_logit_t.npy")
  # next_ks_golden = np.save(f"output/compare/lit_gpt/{step}_{sub_step}_next_ks.npy")
  # next_vs_golden = np.save(f"output/compare/lit_gpt/{step}_{sub_step}_next_vs.npy")
  # assert np.allclose(logits_a, logits_a_golden)
  # assert np.allclose(logit_t, logit_t_golden)
  # assert np.allclose(next_ks, next_ks_golden)
  # assert np.allclose(next_vs, next_vs_golden)

  next_audio_tokens = []
  for logit_a in np.split(logits_a, 7, -1):
    next_a = sample(logit_a, **kwargs)
    next_audio_tokens.append(next_a)
  next_t = sample(logit_t, **kwargs)
  return next_audio_tokens, next_t, next_ks, next_vs


def concat_feat(audio_emb: np.ndarray, input_embs: np.ndarray):
  audio_len = audio_emb.shape[1]
  for i in range(7):
    input_embs[i, 0, 1:audio_len + 1, :] = audio_emb[0, :audio_len].copy()
  return input_embs


def generate_AA(
    audio_features: np.ndarray,
    input_ids: list,
    step,
    adapter: onnxruntime.InferenceSession, wte: onnxruntime.InferenceSession, lit_gpt: onnxruntime.InferenceSession,
    max_returned_tokens: int = 2048,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id_a: Optional[int] = None,
    eos_id_t: Optional[int] = None,
    pad_id_t: Optional[int] = None,
    shift: Optional[int] = None,
    include_prompt: bool = True,
    generate_text=False,
) -> np.ndarray:

  T = input_ids[0].shape[1]

  output = [[] for _ in range(8)]
  # audio_features_golden = np.load(f"output/compare/adapter/audio_features_{step}.npy")
  # assert np.allclose(audio_features_golden, audio_features)
  (audio_embs,) = adapter.run(['audio_embs'], {
      'audio_features': audio_features})  # [1,audio_len,768]
  # audio_embs_golden = np.load(f"output/compare/adapter/audio_embs_{step}.npy")
  # assert np.allclose(audio_embs_golden, audio_embs)

  input_ids: np.ndarray = np.stack(input_ids)  # [8,1,seq_len]
  # input_ids_golden = np.load(f"output/compare/adapter/input_ids_{step}.npy")
  # assert np.allclose(input_ids_golden, input_ids)
  (input_embs, ) = wte.run(None, {'input_ids': input_ids})
  # input_embs_golden = np.load(f"output/compare/adapter/input_embs_{step}.npy")
  # assert np.allclose(input_embs_golden, input_embs)

  input_embs_concat = concat_feat(audio_embs, input_embs)
  # input_embs_concat_golden = np.load(f"output/compare/adapter/input_embs_concat{step}.npy")
  # assert np.allclose(input_embs_concat_golden, input_embs_concat)

  past_ks = np.empty([24, 1, 14, 0, 64], dtype=np.float32)  # 1,14,2048,64
  past_vs = np.empty([24, 1, 14, 0, 64], dtype=np.float32)  # 1,14,2048,64
  tokens_A, token_T, past_ks, past_vs = next_token_A1T2(
      lit_gpt,
      input_embs_concat,
      np.arange(0, T),
      past_ks,
      past_vs,
      step,
      1,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
  )
  for i in range(7):
    output[i].append(tokens_A[i][0].item())
  output[7].append(token_T[0].item())
  input_pos = np.array([T], dtype=np.int64)

  text_end = False
  for sub_step in tqdm(range(2, max_returned_tokens - T + 1)):
    # ring shift
    model_input_ids = []
    for i in range(7):
      model_input_ids.append(
          layershift(tokens_A[i], i)
          .reshape([1, -1])
          .astype(np.int64)
      )
    model_input_ids.append(token_T.reshape([1, -1]).astype(np.int64))
    model_input_ids = np.stack(model_input_ids)
    (input_embs, ) = wte.run(None, {'input_ids': model_input_ids})
    tokens_A, token_T, past_ks, past_vs = next_token_A1T2(
        lit_gpt,
        input_embs,
        input_pos,
        past_ks,
        past_vs,
        step,
        sub_step,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    if text_end:
      token_T = np.array([pad_id_t], dtype=np.int64)

    if tokens_A[-1] == eos_id_a:
      break
    if token_T == eos_id_t:
      text_end = True

    for i in range(7):
      output[i].append(tokens_A[i][0].item())
    output[7].append(token_T[0].item())
    input_pos[0] += 1

  return output


def A1_A2(audio_feature: np.ndarray, input_ids: np.ndarray, leng: int, adapter: onnxruntime.InferenceSession, wte: onnxruntime.InferenceSession, gpt: onnxruntime.InferenceSession, snac: onnxruntime.InferenceSession, text_tokenizer, out_dir, step):
  tokenlist = generate_AA(
      audio_feature,
      input_ids,
      step,
      adapter,
      wte,
      gpt,
      max_returned_tokens=2048,
      temperature=0.9,
      top_k=1,
      eos_id_a=_eoa,
      eos_id_t=_eot,
      pad_id_t=_pad_t,
      shift=padded_text_vocabsize,
      include_prompt=True,
      generate_text=True,
  )
  
  audiolist = reconscruct_snac(tokenlist)
  tokenlist = tokenlist[-1]
  if text_vocabsize in tokenlist:
      tokenlist = tokenlist[: tokenlist.index(text_vocabsize)]
  if out_dir is None:
      out_dir = "./output/default_onnx/A1-A2"
  else:
      out_dir = out_dir + "/A1-A2"
  if not os.path.exists(out_dir):
      os.makedirs(out_dir)
      
  audio = reconstruct_tensors(audiolist)
  (audio_hat,) = snac.run(None, {f'audio_{j}': audio[j].numpy() for j in range(len(audio))})
  sf.write(
      f"{out_dir}/{step:02d}.wav",
      audio_hat.squeeze(),
      24000,
  )

  EXPORT_MODEL = False # for only once.
  return text_tokenizer.decode(torch.tensor(tokenlist)).strip()

def get_time_str():
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return time_str

def main():
  out_dir = f"./output/{get_time_str()}/onnx_infer"
  test_audio_list = sorted(glob.glob('./data/samples/output*.wav'))
  test_audio_transcripts = [
      "What is your name?",
      "what are your hobbies?",
      "Do you like beijing",
      "How are you feeling today?",
      "what is the weather like today?",
  ]

  # load models
  sessOptions = onnxruntime.SessionOptions()
  sessOptions.intra_op_num_threads = 12
  whispermodel = onnxruntime.InferenceSession("output/models/whisper/whisper.onnx", sessOptions)
  sessOptions = onnxruntime.SessionOptions()
  sessOptions.intra_op_num_threads = 12
  apapter = onnxruntime.InferenceSession("output/models/adapter/adapter.onnx", sessOptions)
  sessOptions = onnxruntime.SessionOptions()
  sessOptions.intra_op_num_threads = 12
  wte = onnxruntime.InferenceSession("output/models/wte/wte.onnx", sessOptions)
  sessOptions = onnxruntime.SessionOptions()
  sessOptions.intra_op_num_threads = 12
  lit_gpt = onnxruntime.InferenceSession("output/models/lit_gpt/lit_gpt.onnx", sessOptions)
  sessOptions = onnxruntime.SessionOptions()
  sessOptions.intra_op_num_threads = 12
  snac = onnxruntime.InferenceSession("output/models/snac/snac.onnx", sessOptions)
  text_tokenizer = Tokenizer('checkpoint')

  print("===============================================================")
  print("                       testing A1A2")
  print("===============================================================")
  for step, path in enumerate(test_audio_list):
    mel, leng = load_audio(path)
    audio_feature, input_ids = get_input_ids_whisper(mel, leng, step, whispermodel)
    text = A1_A2(
        audio_feature,
        input_ids,
        leng,
        apapter,
        wte,
        lit_gpt,
        snac,
        text_tokenizer,
        out_dir,
        step,
    )
    print(f"input: {test_audio_transcripts[step]}")
    print(f"output: {text}")
    print(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    )
  print("===============================================================")


if __name__ == '__main__':
  main()
