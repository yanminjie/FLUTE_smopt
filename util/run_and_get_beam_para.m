function  [output_value, data, bunchValid] = run_and_get_beam_para(astra_file_original, astra_file_modified, ...
    params, values, appearances, ...
    out_dist_file, out_LandF_file, output_para)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
current_folder = pwd;

nOutput = length(output_para);
for kk=1:nOutput
    switch output_para{kk}
        case 'PeakCurrent'
        case 'RMSBunchLength'
        case 'FWHMBunchLength'
        case 'MeanEnergy'
        case 'all'
            output_para = {'PeakCurrent', 'RMSBunchLength', 'FWHMBunchLength', 'MeanEnergy'};
            nOutput = length(output_para);
            break
        otherwise
            disp('output beam parameter not supported!')
            return
    end
end

makePlot = 1;
saveDistribution = 1;
lost_threshold = 0.05; % if more than this threshold is lost, data is unvalid.
slicing_para = [5,19];

% ======================================================================
% add the values of the last 3 dipoles
values = horzcat(values, -values(:,end), -values(:,end), values(:,end)); 
params = [params, 'DipoB', 'DipoB', 'DipoB'];
appearances = [appearances, 2, 3, 4];

% write astra runfile
astra_alter_elements(astra_file_original, astra_file_modified, params, values, appearances);
% run astra
idx_rf = strfind(astra_file_modified,'\');
runfile_folder = astra_file_modified(1:idx_rf(end));
idx_of = strfind(astra_file_original,'\');
orifile_folder = astra_file_original(1:idx_of(end));

cd(runfile_folder);
runAstraInMatlab(astra_file_modified);    
cd(current_folder);


bunchValid = 1;
% analyze
if exist(out_dist_file, 'file')
    [dist, bunch_charge, nPart, nPart_lost] = transform_Astra_dist_to_Matlab(out_dist_file);
    LandF = dlmread(out_LandF_file);
    data = analyse_elegant_dist(dist,makePlot,abs(bunch_charge),'DivideBunchLength',slicing_para);

    %  ===========================================
    PeakCurrent = -max(data.current);
    RMSBunchLength = data.sigma_z;
    FWHMBunchLength = data.FWHM_z;
    MeanEnergy = data.gamma_rel * 0.511; %in [MeV]
    if size(LandF,1)>1
        lost_percent = 1-LandF(2,2)/LandF(1,2);
    else
        lost_percent = 1;
    end

    if lost_percent > lost_threshold
        PeakCurrent = NaN;
        RMSBunchLength = NaN;
        FWHMBunchLength = NaN;
        MeanEnergy = NaN;
        bunchValid = 0;        
    end    

    % copy files to the destination
    if saveDistribution
        destination_dist_file = [orifile_folder, 'TrackedDistribution\', out_dist_file(idx_rf(end)+1:end)];
        destination_LandF_file = [orifile_folder, 'TrackedDistribution\', out_LandF_file(idx_rf(end)+1:end)];
        copyfile(out_dist_file, destination_dist_file);
        copyfile(out_LandF_file, destination_LandF_file);

        fid = fopen('history_DistFileNameList.txt','a');
        fprintf(fid,'%s\n',destination_dist_file);
        fclose(fid);  

        dlmwrite('history_OutputInput.txt', [PeakCurrent, RMSBunchLength, FWHMBunchLength, MeanEnergy, values], '-append','delimiter','\t');
    end
else
     PeakCurrent = NaN;
     RMSBunchLength = NaN;
     FWHMBunchLength = NaN;
     MeanEnergy = NaN;
     bunchValid = 0;
     data = NaN;
     
     lost_percent = 1;
end

for kk=1:nOutput
        eval(['output_value(kk)=', output_para{kk}, ';']);
end

disp([num2str(output_value), ' || ', num2str(lost_percent), ' at ',num2str(values(1:end-3))]);

end

