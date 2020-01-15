%% Author Devan Nair
%  Final Update : 27-06-2015 15-47
%     
%%
clc
%addpath(pwd,'Speech');
%cd Speech
clear;
close all;
disp('>> Neural_Network_Trainer')
tic;
%% Audio input
%%Input File

files = ['test1.mp3';'test2.mp3';'test3.mp3';'test4.mp3'];
file_length = size(files);

for j =1:1:file_length(1,1)
filename = files(j,:);
[y,Fs] = audioread(filename);
audiowrite('temp.flac',y,Fs);
delete 'temp.flac'
samples = [10*Fs+1,20*Fs];
t_rec = (samples(2)-samples(1) +1)/Fs;
clear y Fs
[y,Fs] = audioread(filename,samples);


%% Downsampling
downsample_ratio = Fs/(16000);
downsample_ratio = ceil(downsample_ratio);
y_downsampled = downsample(y,downsample_ratio); 
Fs = Fs/downsample_ratio;



%% MFCC Computation
L = length(y_downsampled);
% Hamming
y1 = y_downsampled(:,2).*hamming(length(y_downsampled));

% Pre-Emphasis
pre_emph = [1 0.63];
y1 = filter(1,pre_emph,y1);


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
% i1 = 1;j1 = 0;
% while(it == 0)
%    it = norm(y1_frame(:,i1));
%    if(it == 0)
%        y1_frame = y1_frame(:,(i1+1:l1-j1));
%         j1 = j1+1;
%    end
% end
% clear it i1 j1 

%Frequency Domain Analysis(FFT)
Nfft = 2^nextpow2(samples_per_frame);
Y = fft(y1_frame,Nfft)/(samples_per_frame/2);
f = (Fs/2)*linspace(0,1,Nfft/2 + 1);
Y_periodogram = (Y.*conj(Y))/Nfft;


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
 

clear m k b

%Calculate Filter Bank Energies
H = transpose(H);
Y_periodogram_prime = Y_periodogram((1:max_len),:);
e_pf  = H*(Y_periodogram_prime);


%Cepstral Coefficient Calculation
log_coeff = log(e_pf);
cepstr_coeff = dct(log_coeff);
cepstr_coeff = cepstr_coeff(1:12,:);



for i = 1:1:l1
 train_input(:,(j-1)*l1+i) = cepstr_coeff(:,i);
 train_target(j,(j-1)*l1+i) = 1;
 
end

 

end

clear file_length

%% Neural Network Design

tr.best_tperf = 1;
count = 0 ;
figure;

while(tr.best_tperf > 0.008)

    
hidden_layer_size = 20;
net = patternnet(hidden_layer_size);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

[net,tr] = train(net,train_input,train_target);

plotperform(tr)

if(count >= 5)
    tr.best_tperf = 0.007 ;
end


count = count + 1 ;

end

view(net)

%self_org_net = selforgmap([8 8]);

%if(~ isnan(train_input))

%figure;
%self_org_net = train(self_org_net,train_input);
%view(self_org_net)
%plotsomhits(self_org_net,train_input)

%end

save ('Song_recogniser.mat','net','tr') ;

clear count 
%% Release
 disp('MFCC Successfully Computed');
 disp('Neural Network Implemented');
 disp('Neural Network Successfully Trained');
 elapsed_time = toc;
 soln = ['Estimated Computing Time : ',num2str(elapsed_time),' sec'];
 disp(soln);
 clear elapsed_time soln;
 cd ..
