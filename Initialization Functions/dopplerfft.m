function [doppler_fft] = dopplerfft(range_fft,p)

% windowing for the doppler fft:
if strcmp(p.windowForDopplerFFT, 'y')
    window2 = createWindow(p.w_doppler,p.nChirps);
    window2 = reshape(window2,[1,numel(window2),1]); 
    for channel = 1: p.nChannels
         for samples = 1: size(range_fft,3)
             % Extract the slice of the data cube corresponding to the current channel and chirp
             data_slice = range_fft(channel, : , samples); % making it a 1x32x1
             
             % Multiply the data slice by the window along the third dimension
             windowed_slice = data_slice .* window2;
             
             % Replace the original data slice with the windowed data slice
             range_fft(channel, :, samples) = windowed_slice;
         end
    end
end

doppler_fft = fftshift(fft(range_fft,p.nChirps,2));

end