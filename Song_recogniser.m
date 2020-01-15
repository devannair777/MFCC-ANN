% %% Author Devan Nair
%    Final Update : 27-06-2015 15-47
%     
%%
clc
%addpath(pwd,'Speech');
%cd Speech
clear;
close all;
disp('>> Song_Recogniser');

tic;

%% Audio input

%%Input File

filename ='test1.mp3';
[y,Fs] = audioread(filename);
audiowrite('temp.flac',y,Fs);
delete 'temp.flac'
samples = [50*Fs+1,60*Fs];
t_rec = (samples(2)-samples(1) +1)/Fs;
clear y Fs
[y,Fs] = audioread(filename,samples);


%% Downsampling
downsample_ratio = Fs/(16000);
downsample_ratio = ceil(downsample_ratio);
y_downsampled = downsample(y,downsample_ratio); 
Fs = Fs/downsample_ratio;

%% Plots Time domain
figure;
subplot(3,2,1);
plot((0:(t_rec*Fs*downsample_ratio - 1))*1/(downsample_ratio*Fs),y(:,2));
title('Recorded Signal');
xlabel('t(sec)');
ylabel('Signal Strength');
subplot(3,2,2);
plot((0:(t_rec*Fs - 1))*1/Fs,y_downsampled(:,2));
title('Downsampled Signal');
xlabel('t(sec)');
ylabel('Signal Strength');

%% MFCC Computation
L = length(y_downsampled);
% Hamming
y1 = y_downsampled(:,2).*hamming(length(y_downsampled));

% Pre-Emphasis
pre_emph = [1 0.63];
y1 = filter(5,pre_emph,y1);
subplot(3,2,3);
plot((0:(t_rec*Fs - 1))*1/Fs,y1)
title('After Pre-emphasis & windowing');
xlabel('t(sec)');
ylabel('Signal Strength');

%%Frame Generation

samples_per_frame = ceil(Fs * 0.025) ;
frame_step = ceil(Fs * 0.010);
temp_var = 0;
temp_var2 =1;
while(temp_var <= L)
    for i = 1:1:samples_per_frame;
        if((i+(temp_var2-1)*frame_step) <= L)
        y1_frame(i,temp_var2) = y1((temp_var2-1)*frame_step+i,1);
        end
        temp_var = (i+(temp_var2-1)*frame_step);
    end
    temp_var2 = temp_var2+1;
end

l1 = length(y1_frame);

clear temp_var temp_var2 ;

% it = 0;
% i = 1;j = 0;
% while(it == 0)
%    it = norm(y1_frame(:,i));
%    if(it == 0)
%        y1_frame = y1_frame(:,(i+1:l1-j));
%         j = j+1;
%    end
% end
% clear it i j 
% 
% l1 = length(y1_frame);

%Frequency Domain Analysis(FFT)
Nfft = 2^nextpow2(samples_per_frame);
Y = fft(y1_frame,Nfft)/(samples_per_frame/2);
f = (Fs/2)*linspace(0,1,Nfft/2 + 1);
Y_periodogram = (Y.*conj(Y))/Nfft;
subplot(3,2,4);
plot(f,abs(Y(1:Nfft/2+1,7))) 
title('Single-Sided Amplitude Spectrum of filtered signal(7th Frame)')
xlabel('Frequency (Hz)') 
ylabel('|Y(f)|')
subplot(3,2,5);
plot(f,angle(Y(1:Nfft/2+1,7))) 
title('Phase Spectrum of filtered signal(7th Frame)')
xlabel('Frequency (Hz)') 
ylabel('\Phi (Y(f))')
subplot(3,2,6);
plot(f,Y_periodogram(1:Nfft/2+1,7))
title('Periodogram 1/N(|Y(f)|^2) (7th Frame)')
xlabel('Frequency (Hz)') 
ylabel('|Y(f)|^2')

%Mel-spaced Filter Banks
m1 = 1125*log(1 + 300/700);
m2 = 1125*log(1 + Fs/(2*700));
m = linspace(m1,m2,28);
f1 = 700*(exp(m/1125) - 1);
b = floor((Nfft+1)*f1/Fs);
clear m1 m2 f1
max_len = max(b);

%Filter Bank Design
clear H m
m = 2;
for m =2:1:length(b)-1
k = 1;
while(k <= max(b))
if(k <= b(m) && k >= b(m-1))
    H(k,m-1) = (k-b(m-1))/(b(m)-b(m-1));
else if(k <= b(m+1) && k >= b(m))
    H(k,m-1) = (b(m+1)-k)/(b(m+1)-b(m));
else 
    H(k,m-1) = 0;
    end
end
k = k+1;
end
end
 
figure;
plot(H)
title('Filter Bank Profile');
xlabel('Frequency(bins)');
ylabel('Amplitude');
clear m k b

%Calculate Filter Bank Energies

H = transpose(H);
Y_periodogram_prime = Y_periodogram((1:max_len),:);
e_pf  = H*(Y_periodogram_prime);


%Cepstral Coefficient Calculation
log_coeff = log(e_pf);
cepstr_coeff = dct(log_coeff);
cepstr_coeff = cepstr_coeff(1:12,:);
figure;
plot(cepstr_coeff(:,(1:7)))
title('Cepstral Coefficient (First 7 Frames)');
xlabel('Coefficients')
ylabel('Magnitude')

%% Recognition

%Patternnet Identification

load Song_recogniser.mat
pattern_outputs = net(cepstr_coeff);
temp_var = size(pattern_outputs);
mi6 = ceil(temp_var(1,2)*0.99);
kgb = pattern_outputs(:,1:mi6);
for t = 1:1:temp_var(1,1)

    en(t,1) = norm(kgb(t,:),2);

end
nsa = max(en)*0.9;
cia = find(en > nsa,1);

if(~ isempty(cia))
    disp ('Song Tested was');
    soln = ['test',num2str(cia),'.mp3'] ;
else
    
    soln = 'Song not in trained sample' ;
end 
disp(soln);

clear temp_var mi6 kgb mi6 cia nsa soln

%Self Organizing Map Identification

%self_org_outputs = self_org_net(cepstr_coeff);

 %% Release
 disp('MFCC Successfully Computed');
 elapsed_time = toc;
 disp('Estimated Computing Time');
 disp(elapsed_time);
 clear elapsed_time;
 cd ..
