function [w] = createWindow(window_type, window_length)
% Takes in the radar parameters and creates a window function
    w = strcmp(window_type,["hamming", "blackman", "hann"]);
    if w(1)
        w = hamming(window_length).';
    elseif w(2)
        w = blackman(window_length).';
    elseif w(3)
        w = hann(window_length).';
    end
end

