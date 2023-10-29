a = 1 ;
b = 51 ;
c = 1 ;
d = 1601; % change
 
p = processData();

for i = 1:149
    
    p.range_ticks = p.range_res*(0:p.nChirps);
    p.doppler_ticks = p.doppler_res*(-1*floor((p.nChannels/2)): floor((p.nChannels/2)));
    
    rti = p.plotrti(a:b, 90:112);
    
    figure1 = figure;
    imagesc(20*log10(abs(rti.')))
    title('Range Time Map')
    colorbar; 

    figure2 = figure;
    imagesc(p.frequency_time(:,c:d))
    title('Spectogram')
    colorbar;

    saveDirectory = 'C:\Users\Acer\Desktop\Uni\sem2\EEE4022S\Signal Processing\Matlab\processedData';
    currentDateTime = datetime('now', 'Format', 'yyyyMMddHHmmss');
    
    filename1 = fullfile(saveDirectory, ['rtm_', num2str(i+99), '.png']);
    filename2 = fullfile(saveDirectory, ['spec_', num2str(i+99), '.png']);
    
    saveas(figure1, filename1);
    saveas(figure2, filename2);
    close all;

    a = a + 50;
    b = b + 50;
    c = c + 1600;
    d = d + 1600;

    disp(i);

end