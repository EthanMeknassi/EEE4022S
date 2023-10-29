function [ram_fft] = ramfft(doppler_fft,p)

%%% windowing for the range fft

    if strcmp(p.windowForAngleFFT, "y")
        window3 = createWindow(p.w_ram,p.nChannels);
        window3 = reshape(window3,[1,numel(window3),1]); %1x32x1
        for chirp = 1: size(doppler_fft,2)
             for samples = 1: size(doppler_fft,3)
                 data_slice = doppler_fft(:, chirp , samples); % making it a 12x1
                 windowed_slice = data_slice .* window3.';
                 doppler_fft(:, chirp, samples) = windowed_slice;
             end
        end
    end

    ram_fft = fftshift(fft(doppler_fft,p.nChannels,1));


end