Collection of Speech Enhancement Models
========================================

* Baseline 1: is a dully connected deep neural network (DNN) to map log-spectral power frames (from STFT)
into log-spectral power frames. This can be used for denoising, or even to recover lost parts
of spectrum (although phase is not processed, it will be left as is). The baseline can be trained with 
an instruction like:

```
python -u train_baseline1.py --batch_size 32 --save_path <ckpt_path> --in_frames 7 --cache_path data/cache --dataset data --patience 10 --cuda --num_workers 2 --save_freq 50
```

TODO: define a bit more training options and extend new baselines, like SEGAN.
