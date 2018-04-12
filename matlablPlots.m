%% Generated with one octave
% dir: corpus/oneSin2
% 16000 samples long 1(s)
% f = [440, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99]
figure;
[x1,fs] = audioread('corpus/scale/sinus0.wav'); 
subplot(2,1,1); plot(x1); 
[y1,fs] = audioread('oneOctave.wav');
subplot(2,1,2); plot(y1);

%% Generated with one sinusoid
figure;
[x2,fs] = audioread('corpus/oneSin/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y2,fs] = audioread('longSin.wav');
subplot(2,1,2); plot(y2);

%% Generated with quantization channels = 8
% tensorflow-wavenet aleix$ python train.py --data_dir=corpus/oneSin --num_steps=100 --silence_threshold=0.0000001
% python generate.py --samples 10000 logdir/train/2018-03-15T23-29-59/model.ckpt-99 --wav_out_path=quantization8.wav
figure;
[x2,fs] = audioread('corpus/oneSin/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y3,fs] = audioread('quantization8.wav');
subplot(2,1,2); plot(y3);

%% Generated with quantization channels = 32
% python train.py --data_dir=corpus/oneSin --num_steps=100 --silence_threshold=0.0000001
% python generate.py --samples 10000 logdir/train/2018-03-16T00-14-28/model.ckpt-99 --wav_out_path=quantization32.wav
figure;
[x2,fs] = audioread('corpus/oneSin/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y4,fs] = audioread('quantization32.wav');
subplot(2,1,2); plot(y4);

%% Generated with quantization channels = 64
% quantization64.wav logdir/train/2018-03-21T17-08-02/model.ckpt-99
figure;
[x2,fs] = audioread('corpus/oneSin/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y5,fs] = audioread('quantization64.wav');
subplot(2,1,2); plot(y5);

%% Generated with quantization channels = 128
figure;
[x2,fs] = audioread('corpus/oneSin/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y5,fs] = audioread('quantization128.wav');
subplot(2,1,2); plot(y5);
%GOOD

%% Generated with dilations = 1
% python train.py --data_dir=corpus/oneSin --num_steps=100 --silence_threshold=0.000000
% python generate.py --samples 10000 logdir/train/2018-03-16T05-03-37/model.ckpt-99 --wav_out_path=dilations1.wav
figure;
[x2,fs] = audioread('corpus/oneSin/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y4,fs] = audioread('dilations1.wav');
subplot(2,1,2); plot(y4);
%% Generated with dilations = 1, 2, 4, 8
%python train.py --data_dir=corpus/oneSin --num_steps=100 --silence_threshold=0.0000001
figure;
[x2,fs] = audioread('corpus/oneSin/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y5,fs] = audioread('dilations1x1248.wav');
subplot(2,1,2); plot(y5);

%% Generated with dilations = 1,2,4,8, 16, 32, 64, 128, 256, 512
figure;
[x2,fs] = audioread('corpus/oneSin/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y6,fs] = audioread('dilations512.wav');
subplot(2,1,2); plot(y6);

%% Generated with dilations = 1,2,4,8, 16, 32, 64, 128, 256, 512, 1,2,4,8, 16, 32, 64, 128, 256, 512
% python train.py --data_dir=corpus/oneSin --num_steps=100 --silence_threshold=0.0000001
%on macbook
figure;
[x2,fs] = audioread('corpus/oneSin/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y7,fs] = audioread('dilations512512.wav');
subplot(2,1,2); plot(y7);
%GOOD!
%% Generated with dilations = 5x(1,2,4,8)
%16/03 09:29
%python train.py --data_dir=corpus/oneSin --num_steps=100 --silence_threshold=0.0000001
%python generate.py --samples 10000 --wav_out_path=dilations5x1248.wav logdir/train/2018-03-16T09-22-15/model.ckpt-99
figure;
[x2,fs] = audioread('corpus/oneSin/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y8,fs] = audioread('dilations5x1248.wav');
subplot(2,1,2); plot(y8);
%GOOD!

%% Generated with dilations = 2x(1,2,4,8)
%./logdir/train/2018-03-16T09-34-27
figure;
[x2,fs] = audioread('corpus/oneSin/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y9,fs] = audioread('dilations2x1248.wav');
subplot(2,1,2); plot(y9);

%% %% Generated with dilations = 3x(1,2,4,8)
%2018-03-21T17-08-02/model.ckpt-99
figure;
[x2,fs] = audioread('corpus/oneSin/sinus16000.wav');
subplot(2,1,1); plot(x2); 
[y9,fs] = audioread('dilations3x1248.wav');
subplot(2,1,2); plot(y9);
%% Reduced batch size

%% Conditionality
figure;
[y10,fs] = audioread('conSin1.wav');
plot(y10)
figure;
[y11,fs] = audioread('conSin2.wav');
plot(y11)

%% Calculate frequencies
figure;
[y10,fs] = audioread('conSin1.wav');
plot(y10)
hold on;
x_1 = find(y10 == 1);
x_2 = find(y10 == -1);
x_pos1= x_1;
y_pos1=y10(x_1);
x_pos2= x_2;
y_pos2=y10(x_2);
plot(x_pos1(length(x_pos1)),y_pos1(length(y_pos1)),'r*')
hold on;
plot(x_pos2(length(x_pos2)),y_pos2(length(y_pos2)),'r*')
freq = abs(1/(2*((x_pos1(length(x_pos1))-x_pos2(length(x_pos2)))))*16000);
disp(freq);
%%
figure;
[y11,fs] = audioread('conSin2.wav');
plot(y11)
hold on;
x_1 = find(y11 == 1);
x_2 = find(y11 == -1);
x_pos1= x_1;
y_pos1=y11(x_1);
x_pos2= x_2;
y_pos2=y11(x_2);
plot(x_pos1(length(x_pos1)),y_pos1(length(y_pos1)),'r*')
hold on;
plot(x_pos2(length(x_pos2)),y_pos2(length(y_pos2)),'r*')
freq = abs(1/(2*((x_pos1(length(x_pos1))-x_pos2(length(x_pos2)))))*16000);
disp(freq);

%% Conditional Scale
figure;
[y11,fs] = audioread('scaleSin6.wav');
plot(y11)
hold on;
x_1 = find(y11 == 1);
x_2 = find(y11 == -1);
x_pos1= x_1;
y_pos1=y11(x_1);
x_pos2= x_2;
y_pos2=y11(x_2);
plot(x_pos1(length(x_pos1)),y_pos1(length(y_pos1)),'r*')
freq = abs(1/(2*((x_pos1(length(x_pos1))-x_pos2(length(x_pos2)))))*16000);
disp(freq);

%% Shapes
figure;
[y12,fs] = audioread('generatedSignals/shape440.wav');
plot(y12)

figure;
[y13,fs] = audioread('generatedSignals/shape880.wav');
plot(y12)

%figure
%Nx = length(y12);
%nsc = floor(Nx/4.5);
%nov = floor(nsc/2);
%nff = max(256,2^nextpow2(nsc));
%spectrogram(y12,hamming(nsc),nov,nff);
%maxerr = max(abs(abs(t(:))-abs(s(:))))
%spectrogram(y12,256,250,256,1e3,'yaxis')

%% Local Conditioning on Generate
[y13,fs] = audioread('amajor.wav');
%plot(y13)
%soundsc(y13)
length(y13)





