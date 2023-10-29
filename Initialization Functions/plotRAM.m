function [] = plotRAM(ram_fft, doppler_fft, p)
    
    N = size(doppler_fft(:,:,3),1);
    M = size(doppler_fft(:,:,3),2);
    p.range_ticks = p.range_res*(0:M);
    p.doppler_ticks = p.doppler_res*(-1*floor((N/2)): floor((N/2)));

    if strcmp(p.ChirpSum, "y")
            
        plottingram = sum(ram_fft,2); 
        ram = reshape(plottingram, p.nChannels, size(doppler_fft,3) , []);    
    
    elseif strcmp(p.ChirpSum, "n")
           
        ram = reshape(ram_fft(:,p.ChirpToOutput,:),p.nChannels,p.nSamples,[]);  
    
    end
    
    figure();
    imagesc(20*log10(abs(ram)));
    xlabel('Angle');
    ylabel('Range');
    title('Range-Angle Map');
    
end