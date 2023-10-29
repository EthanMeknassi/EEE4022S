function [range_fft] = rangefft(data_cube,p)

%%% windowing for the range fft

if strcmp(p.windowForRangeFFT, "y")
    window = createWindow(p.w_range,p.nSamples);
    window = reshape(window,[1,1,numel(window)]); %1x1x96
    for channel = 1: p.nChannels
         for chirp = 1: p.nChirps
             % Extract the slice of the data cube corresponding to the current channel and chirp
             data_slice = data_cube(channel, chirp , :); % making it a 1x1x96
             
             % Multiply the data slice by the window along the third dimension
             windowed_slice = data_slice .* window;
             
             % Replace the original data slice with the windowed data slice
             data_cube(channel, chirp, :) = windowed_slice;
         end
    end
end

range_fft = fft(data_cube,p.nSamples,3);

end