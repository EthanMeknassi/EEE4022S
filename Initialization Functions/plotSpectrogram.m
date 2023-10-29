function [] = plotSpectrogram(range_fft, p)

    bindata = reshape(range_fft(:,:,p.range_bin), p.nChannels, p.nChirps, []);

    if strcmp(p.SpecChannelSum, 'y')
        summedchans = sum(bindata,1); %uncomment for summed channels
    else
        summedchans = reshape(bindata(p.ChannelToOutput,:),p.nChirps,[]).'; % uncomment for uunsummed channles
    end

    num_windows = floor((length(summedchans) - p.overlap * p.window_size) / (p.window_size - p.overlap * p.window_size));
    spec = zeros(p.window_size * p.padding, num_windows);

    for i = 1:num_windows
   
        start_idx = (i - 1) * (p.window_size - p.overlap * p.window_size) + 1;
        end_idx = start_idx + p.window_size - 1;
        range_window = summedchans(start_idx:end_idx);
    
        win = createWindow(p.w_spec,p.window_size);
    
        range_window = range_window .* win;
      
        spec(:, i) = fftshift(fft(range_window,p.window_size*p.padding,2));
    end
    
    figure();
    imagesc(20*log10(abs(spec)));
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Spectrogram');


end