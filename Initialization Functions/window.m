function [window] = window(window_type, window_length)
% Takes in the radar parameters and creates a window function
    w = strcmp(window_type,["hamming", "blackman", "hann"]);
    if w(1)
        window = hamming(window_length).';
    elseif w(2)
        window = blackman(window_length).';
    elseif w(3)
        window = hann(window_length).';
    end
end
