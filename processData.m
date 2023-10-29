function [p] = processData()
    
    %clear all; clc;
    warning('off');
    % warning('on');
    addpath('.\Initialization Functions\')
    addpath('.\Raw HDF5 Data\')
    addpath('.\readData\')
    
    %% ------------------ Parameters Setup --------------------------
    filename = 'C:\Users\Acer\Desktop\Uni\sem2\EEE4022S\Signal Processing\Matlab\data\Ethans Data\Hand_Gestures_Waving2.hdf5'; % REPLACE WITH DATA FILE
    p = get_parameters(filename);
    p.frameToProcess = 300 ;
    
    %% -------------------  RDM Parameters --------------------------
    
    p.w_range = 'hann'; % hamming, blackman, hann, none
    p.w_doppler = 'hann'; % hamming, blackman, hann, none
    p.windowForRangeFFT = 'y'; % y or n 
    p.windowForDopplerFFT = 'y'; % y or n 
    p.ChannelSum = 'n';  % y or n
    
    
    %% ----------------- Spectogram parameters ----------------------
    p.w_spec = 'hann'; % hamming, blackman, hann, none
    p.range_bin = [ 108 109 ]; % Must be an array
    p.window_size = 30;
    p.overlap = 0.99;
    p.padding = 10;
    p.ChannelToOutput = 1 ;
    p.SpecChannelSum = 'n';
    
    %% ------------------- Data Processing ---------------------------

    data_cubes = zeros([p.nChannels, p.nFrames, p.nChirps, p.nSamples]);
    p.Frameprocessed = 1;
    
    for i = 1:p.nFrames-1
    
        p.frameToProcess = i;
        data_cube = constructCube(p);
        data_cubes(:,i,:,:) = data_cube;
        
        window = createWindow(p.w_range,p.nSamples);
        window = reshape(window,[1,1,1,numel(window)]);
        for channel = 1: p.nChannels
             for chirp = 1: p.nChirps
                 %Extract the slice of the data cube corresponding to the current channel and chirp
                 data_slice = data_cubes(channel, i, chirp , :); 
                 
                 %Multiply the data slice by the window along the third dimension
                 windowed_slice = data_slice .* window;
                 
                 %Replace the original data slice with the windowed data slice
                 data_cubes(channel, i, chirp, :) = windowed_slice;
             end
        end
    
    end
    
    averaged_slices = zeros(size(data_cubes, 1), size(data_cubes, 3), size(data_cubes, 4));
    
    for i = 1:10
        averaged_slices = averaged_slices + squeeze(data_cubes(:, i, :, :));
    end
    
    averaged_slices = averaged_slices / 10;
    reshaped_array = reshape(averaged_slices, [size(averaged_slices, 1), 1, size(averaged_slices, 2), size(averaged_slices, 3)]);
    final_array = repmat(reshaped_array, [1, p.nFrames, 1, 1]);
    
    data_cubes = data_cubes - final_array;
    
    data_cubes = squeeze(sum(data_cubes,1));
    % data_cubes = squeeze(data_cubes(1,:,:,:));
    
    range_fft = fft(data_cubes, p.nSamples,3);
    
    p.plotrti = range_fft(:,1,:);
    p.plotrti = squeeze(p.plotrti);
    
    %% ------------------------------------ RDM -----------------------------------------------------
    
    window2 = createWindow(p.w_doppler,p.nChirps);
    window2 = reshape(window2,[1,numel(window2),1]); 
    for i = 1:size(range_fft,1)
         for samples = 1: size(range_fft,3)
             % Ext  ract the slice of the data cube corresponding to the current channel and chirp
             data_slice = range_fft(i, : , samples); 
                
             % Multiply the data slice by the window along the third dimension
             windowed_slice = data_slice .* window2;
                           
             % Replace the original data slice with the windowed data slice
             range_fft(i, :, samples) = windowed_slice;
         end
     end
    
    doppler_fft = fftshift(fft(range_fft,p.nChirps,2));
    doppler_fft = squeeze(doppler_fft(40,:,:)); % CHOOSE FRAME TO OUTPUT
    figure();
    imagesc(flip(p.range_ticks), p.doppler_ticks, 20*log10(abs(doppler_fft)))
    xlabel("Range [m]")
    ylabel("Doppler [Hz]")
    
    title('Range-Doppler Map');
    
    
    
    %% --------------------- DTM ---------------------------
    
       microDopplerMatrix = zeros(p.nChirps,p.nFrames);
       
       for n = 1:p.nFrames-1
   
           p.frameToProcess = n;
           frame = constructCube(p);
           frame = squeeze(frame(1,:,:));
           
           if sum(strcmp(p.w_range,["hamming","blackman","hann"]))
               w = createWindow(p.w_range, p.nSamples);
               for i = 1:p.nChirps
                  frame(i,:) = frame(i,:).*w;
               end
           end

           range_fft = fft(frame,p.nSamples,2);
           microDopplerMatrix(:,n) = [sum(range_fft(:,p.range_bin),2)];
       end

       microDopplerVector = reshape(microDopplerMatrix,1,p.nChirps*p.nFrames);

       shift = floor(p.window_size*(1-p.overlap));
       if shift < 1
           shift = 1;
       end

       frequency_time = zeros(p.padding + p.window_size,ceil((p.nChirps*p.nFrames-p.window_size)/shift));
       

       i = 1;
       for n = 1:shift:p.nChirps*p.nFrames-p.window_size
           microDopplerWindow = microDopplerVector(n:n+p.window_size-1);
           if sum(strcmp(p.w_spec,["hamming","blackman","hann"]))
               w = createWindow(p.w_spec, length(microDopplerWindow));
               microDopplerWindow = microDopplerWindow.*w;
           end
           microDopplerWindow = [microDopplerWindow zeros(1,p.padding)];
           frequency_time(:,i) = fftshift(fft(microDopplerWindow));
           i = i + 1;

       end
   
   p.frequency_time = 20*log10(abs(frequency_time))-max(max(20*log10(abs(frequency_time))));

   

end