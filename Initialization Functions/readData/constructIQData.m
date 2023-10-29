% This is a revised code from William Bourn's implementation in 2022. 


function[output_data] = constructIQData(input_data, p)

    input_size = size(input_data,2);
    
    %Define the empty output array.
    output_data = complex(zeros(2,4096));
    
    %Compensate for sign extension in ADC where the sample size is less
    %than 16 bits
    if p.nBits < 16
        max = 2^(p.nBits -1) -1;
        input_data(input_data > max) = input_data(input_data > max) - 2^p.nBits;
    end
    
    %Reshape the raw data into a 4xN matrix
    input_data = double(reshape(input_data, 4, []));
    
    %Combine the four elements of each column into two complex values
    for x = 1: input_size/4
        output_data(1, x) = input_data(1, x) + 1j*input_data(3, x);
        output_data(2, x) = input_data(2, x) + 1j*input_data(4, x);
    end
    
    %Reshape the output data into an array
    output_data = reshape(output_data, 1, []);
%     disp(size(output_data));
    %----------------------------------------------------------------------
end

%--------------------------------------------------------------------------
