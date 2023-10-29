function[p] = get_parameters(filename)

    
    p.filename = filename;

%% --------Important parameters--------
    info = h5info(filename);
    p.nFrames =  length(info.Groups(2).Groups);  % Number of chirps per frame
    p.nChirps = double(h5read(filename, '/Parameters/frameCfg/numChirps'));  % Total Number of Chiprs Stored
    p.nSamples = double(h5read(filename, '/Parameters/profileCfg/numAdcSamples'));  % Number of samples per chirp
    
    %% ----------- Getting the number of channels: ---------
    recBitMask = dec2bin(h5read(filename, '/Parameters/channelCfg/rxChannelEn'), 8); % Read and convert to binary
    
    numRx = 0;
    for bitIdx = 1:length(recBitMask)
        bit = recBitMask(bitIdx);
        numRx = numRx + str2double(bit); % Count the number of '1' bits in the binary mask
    end
    
    chirpEndIndex = h5read(filename, '/Parameters/frameCfg/chirpEndIndex');
    chirpStartIndex = h5read(filename, '/Parameters/frameCfg/chirpStartIndex');
    
    numTx = chirpEndIndex - chirpStartIndex + 1;
    p.nChannels = numTx * numRx;

%% -------Intermediate/Useful parameters-------

    p.nBits = 16; % 16 Bit adc - constant value
    p.bytespersample = 2;

    p.real_or_complex = mod((double(h5read(filename, '/Parameters/adcbufCfg/adcOutputFmt/')) + 2),3)  ;   % real or complex (1 being real and 2 being complex)
    p.frequency_slope = double(h5read(filename, '/Parameters/profileCfg/freqSlopeConst')) * 1e12; % frequency Slope [Hz/s]
    p.time_sweep = double(h5read(filename, '/Parameters/profileCfg/rampEndTime')) * 1e-6 ;  % Ramp Time [s]
    p.time_idle = double(h5read(filename, '/Parameters/profileCfg/idleTime')) *1e-6 ;  % Time idle between  [s]
    p.sampling_rate = double(h5read(filename, '/Parameters/profileCfg/digOutSampleRate')) *1e3  ;  % Sampling rate in [sps]
    p.frequency_start = double(h5read(filename, '/Parameters/profileCfg/startFreq')) * 1e9;  % Starting frequency of the radar [Hz]
    p.frame_period = double(h5read(filename, '/Parameters/frameCfg/framePeriod'))* 1e-3; %  Frame Period [s]
    
    p.frame_size = p.nChirps*p.nSamples*2*p.real_or_complex*p.nChannels;  % Number of bytes per frame [Bytes]
    p.bandwidth = (p.nSamples/p.sampling_rate)*p.frequency_slope;  % Sampled bandwidth [hz]
    p.time_chirp = p.time_idle + p.time_sweep;  % Total chirp time (idle time + ramp time) [s]
    p.frequency_centre = (p.frequency_start + (p.frequency_slope*p.time_sweep)/2) ;  % Center frequency of chirp (calculated from total bandwidth) [Hz]
%% -------Performance metrics-------

    p.range_max = 3e8*p.sampling_rate/(2*p.frequency_slope); % Maximum unambigious range [m]
    p.range_res = 3e8/(2*p.bandwidth);  % Range resolution [m]
    p.doppler_res = 1/(p.nChirps*p.time_chirp);  % Doppler resolution [Hz]
    p.velocity_max = ((3e8/(4*p.frequency_centre*p.time_chirp)));  % Maximum unambigious velocity [m/s]
    p.velocity_res = p.doppler_res*((3e8/(2*p.frequency_centre)));  % Velocity resolution [m/s]

end
