%%% Requirements:
%%% psignifit 4: https://github.com/wichmann-lab/psignifit/wiki 
%%% ---

function calculateTAEmagnitudes(participant_list)
    clc; clear;

    TAE_mags = {'uN1', 'uN2', 'uC', 'uF1', 'uF2',...
            'cN1', 'cN2','cC','cF1', 'cF2',...
            'dN1', 'dN2', 'dC', 'dF1',  'dF2'};


    %% Row index for data storage
    row_index = 2;

    for participant = participant_list  
        %% Get the data
        no_adap_file_path = ['../experimental_data/', num2str(participant), '_No_adaptation.csv'];
        adap_file_path = ['../experimental_data/', num2str(participant), '_Adaptation.csv'];
        no_adap_data_cell = readcell(no_adap_file_path);
        adap_data_cell = readcell(adap_file_path);

        %% Extract the needed data for threshold calculation [x_location, y_location, location_name, orientation, test key]
        no_adap_data = no_adap_data_cell(:, [2, 3, 4, 5, 6]);
        adap_data = adap_data_cell(:, [2, 3, 4, 5, 6]);

        %% Init location names
        locations = {'uN1', 'uN2', 'uC', 'uF1', 'uF2',...
            'cN1', 'cN2','cC','cF1', 'cF2',...
            'dN1', 'dN2', 'dC', 'dF1',  'dF2'};

        %% Go through locations
        for l = 1:length(locations)

            %% Get the location specific data
            location = locations{l}
            no_adap_temp_location_data = no_adap_data(strcmp(no_adap_data(:, 2), location), :);
            adap_temp_location_data = adap_data(strcmp(adap_data(:, 2), location), :);

            %% Get stimulus instensities
            no_adap_stim_temp_level = unique([no_adap_temp_location_data{2:end, 3}]);
            adap_stim_temp_level = unique([adap_temp_location_data{2:end, 3}]);

            %% Calculate number of positive responses (e.g., perceived clockwise tilt:
            %%'right' arrow key button) at each of the stimulus temp_level
            no_adap_n_right_key = [];
            adap_n_right_key = [];

            % Go over stimulus temp_levels
            for i = 1:length(no_adap_stim_temp_level)
                no_adap_temp_level = no_adap_stim_temp_level(i);
                no_adap_responses = {no_adap_temp_location_data{2:end, 5}};
                no_adap_truth_array = no_adap_temp_level == [no_adap_temp_location_data{2:end, 3}];
                no_adap_resps_temp_level = {no_adap_responses{no_adap_truth_array}};
                no_adap_N = nnz(strcmp(no_adap_resps_temp_level, 'right'));
                no_adap_n_right_key = [no_adap_n_right_key, no_adap_N];
            end

            for i = 1:length(adap_stim_temp_level)
                adap_temp_level = adap_stim_temp_level(i);
                adap_responses = {adap_temp_location_data{2:end, 5}};
                adap_truth_array = adap_temp_level == [adap_temp_location_data{2:end, 3}];
                adap_resps_temp_level = {adap_responses{adap_truth_array}};
                adap_N = nnz(strcmp(adap_resps_temp_level, 'right'));
                adap_n_right_key = [adap_n_right_key, adap_N];
            end

            %% Calculate the number of trials at each stimulus temp_level
            no_adap_ntotal = [];
            adap_ntotal = [];

            for i = 1:length(no_adap_stim_temp_level)
                no_adap_temp_level = no_adap_stim_temp_level(i);
                no_adap_n = sum(no_adap_temp_level == [no_adap_temp_location_data{2:end, 3}]);
                no_adap_ntotal = [no_adap_ntotal, no_adap_n];
            end

            for i = 1:length(adap_stim_temp_level)
                adap_temp_level = adap_stim_temp_level(i);
                adap_n = sum(adap_temp_level == [adap_temp_location_data{2:end, 3}]);
                adap_ntotal = [adap_ntotal, adap_n];
            end
            
            %% Init Options for psychometric function
            options = struct;
            options.sigmoidName = 'logistic';
            options.exp.Type = 'equalAsymptote';
            options.fixedPars = [NaN, NaN, .01, 0, 0]; % (threshold, width, lapse, guess and eta)

            %% Fit the results
            no_adap_fit_data = [transpose(no_adap_stim_temp_level), transpose(no_adap_n_right_key), transpose(no_adap_ntotal)];
            no_adap_results = psignifit(no_adap_fit_data, options);
            no_adap_temp_threshold = no_adap_results.Fit(1);

            adap_fit_data = [transpose(adap_stim_temp_level), transpose(adap_n_right_key), transpose(adap_ntotal)];
            adap_results = psignifit(adap_fit_data, options);
            adap_temp_threshold = adap_results.Fit(1);
            
            
            %% Store the TAE_mags
            %find the column index of the location: no_adap and adap
            no_adap_col = strcat('no_adap_', location);
            no_adap_col_index = find(contains(TAE_mags(1,:), no_adap_col));
            
            adap_col = strcat('adap_', location);
            adap_col_index = find(contains(TAE_mags(1,:), adap_col));
            %add the value at the end of the column
            TAE_mags{row_index, no_adap_col_index} = adap_temp_threshold - no_adap_temp_threshold;
        end
        row_index = row_index + 1;
    end

    %% Save the data as .csv file
    filename = './TAE_magnitudes.csv';
    T = cell2table(TAE_mags(2:end,:),'VariableNames', TAE_mags(1,:));
    
    % Write the table to a CSV file
    writetable(T,filename);

end