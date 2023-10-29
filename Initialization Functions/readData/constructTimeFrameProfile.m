% This is a revised code from William Bourn's implementation in 2022. 

function[output_data] = constructTimeFrameProfile(input_data, p)

    
    %Determine the size of the input data array
    input_size = size(input_data, 2);
    
    %Determine whether the input data has the correct number of values
    if input_size ~= p.nChannels*p.nChirps*p.nSamples
        %Throw error
        error("Error: Data array does not contain the correct number of samples.")
    end
    
    %Define the empty output array
    output_data = complex(zeros(p.nChannels, 1, p.nChirps, p.nSamples));
    
    %Reshape the unordered data into a 4-dimensional array
    input_data = reshape(input_data, p.nSamples, p.nChannels, p.nChirps, []);
    
    %Rearrange the indices
    for w = 1:p.nChannels
        for x = 1:1
            for y = 1:p.nChirps
                for z = 1:p.nSamples
                    output_data(w,x,y,z) = input_data(z,w,y,x);
                end
            end
        end
    end
    
    %----------------------------------------------------------------------
end

%--------------------------------------------------------------------------
