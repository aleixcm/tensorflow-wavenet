%% Generated with one octave
% dir: longersinus2
% 16000 samples long 1(s)
% f = [440, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99]
figure;
[x1,fs] = audioread('longerSinus2/sinus0.wav'); 
subplot(2,1,1); plot(x1); 
[y1,fs] = audioread('oneOctave.wav');
subplot(2,1,2); plot(y1);

%% Generated with one sinusoid
figure;
[x2,fs] = audioread('longerSinus/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y2,fs] = audioread('longSin.wav');
subplot(2,1,2); plot(y2);

%% Generated with quantization channels = 8
% tensorflow-wavenet aleix$ python train.py --data_dir=longerSinus --num_steps=100 --silence_threshold=0.0000001
% python generate.py --samples 10000 logdir/train/2018-03-15T23-29-59/model.ckpt-99 --wav_out_path=quantization8.wav
figure;
[x2,fs] = audioread('longerSinus/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y3,fs] = audioread('quantization8.wav');
subplot(2,1,2); plot(y3);

%% Generated with quantization channels = 32
% python train.py --data_dir=longerSinus --num_steps=100 --silence_threshold=0.0000001
% python generate.py --samples 10000 logdir/train/2018-03-16T00-14-28/model.ckpt-99 --wav_out_path=quantization32.wav
figure;
[x2,fs] = audioread('longerSinus/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y4,fs] = audioread('quantization32.wav');
subplot(2,1,2); plot(y4);

%% Generated with quantization channels = 128
figure;
[x2,fs] = audioread('longerSinus/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y5,fs] = audioread('quantization128.wav');
subplot(2,1,2); plot(y5);

%% Generated with dilations = 1
% python train.py --data_dir=longerSinus --num_steps=100 --silence_threshold=0.000000
% python generate.py --samples 10000 logdir/train/2018-03-16T05-03-37/model.ckpt-99 --wav_out_path=dilations1.wav
figure;
[x2,fs] = audioread('longerSinus/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y4,fs] = audioread('dilations1.wav');
subplot(2,1,2); plot(y4);
%% Generated with dilations = 1, 2, 4, 8
%python train.py --data_dir=longerSinus --num_steps=100 --silence_threshold=0.0000001
figure;
[x2,fs] = audioread('longerSinus/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y5,fs] = audioread('dilations8.wav');
subplot(2,1,2); plot(y5);

%% Generated with dilations = 1,2,4,8, 16, 32, 64, 128, 256, 512
figure;
[x2,fs] = audioread('longerSinus/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y6,fs] = audioread('dilations512.wav');
subplot(2,1,2); plot(y6);

%% Generated with dilations = 1,2,4,8, 16, 32, 64, 128, 256, 512, 1,2,4,8, 16, 32, 64, 128, 256, 512
% python train.py --data_dir=longerSinus --num_steps=100 --silence_threshold=0.0000001
%on macbook
figure;
[x2,fs] = audioread('longerSinus/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y7,fs] = audioread('dilations512512.wav');
subplot(2,1,2); plot(y7);
%GOOD!
%% Generated with dilations = 5x(1,2,4,8)
%16/03 09:29
%python train.py --data_dir=longerSinus --num_steps=100 --silence_threshold=0.0000001
%python generate.py --samples 10000 --wav_out_path=dilations5x1248.wav logdir/train/2018-03-16T09-22-15/model.ckpt-99
figure;
[x2,fs] = audioread('longerSinus/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y8,fs] = audioread('dilations5x1248.wav');
subplot(2,1,2); plot(y8);
%GOOD!

%% Generated with dilations = 2x(1,2,4,8)
%./logdir/train/2018-03-16T09-34-27
figure;
[x2,fs] = audioread('longerSinus/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y9,fs] = audioread('dilations2x1248.wav');
subplot(2,1,2); plot(y9);

%% Reduced batch size




