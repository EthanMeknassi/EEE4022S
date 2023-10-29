function [] = plotRDM(doppler_fft, range_fft, p)


    N = size(doppler_fft(:,:,:),1);
    M = size(doppler_fft(:,:,:),2);
    p.range_ticks = p.range_res*(0:M);
    p.doppler_ticks = p.doppler_res*(-1*floor((N/2)): floor((N/2)));


    if strcmp(p.ChannelSum, "y")
            
        plottingdop = sum(doppler_fft,1);
        rdm = reshape(plottingdop(1,:,:), p.nChirps, size(range_fft,3) , []);
    
    elseif strcmp(p.ChannelSum, "n")
           
        rdm = reshape(doppler_fft(p.ChannelToOutput,:,:), p.nChirps, size(range_fft,3) , []);
    
    end



    % figure1 = figure;
    figure();
    imagesc(flip(p.range_ticks), p.doppler_ticks, 20*log10(abs(rdm)))
    title('Range Doppler Map')
    xlabel('Range [m]')
    ylabel('Doppler [Hz]')
    colorbar; 

end