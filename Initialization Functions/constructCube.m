function [data_cube] = constructCube(p)

    frame_name = ['/Data/Frame_', num2str(p.frameToProcess) ,'/frame_data'];
    frame = h5read(p.filename, frame_name).';

    data_frame = constructIQData(frame, p);
    data_cube = constructTimeFrameProfile(data_frame, p);
    
    data_cube = reshape(data_cube(:,1,:,:), p.nChannels, p.nChirps, p.nSamples, []);

end