function fitness = ga_runSimulationFcn_parallel(x)

% nVargin = nargin - 1;
% if nVargin > 0
%     field1 = 'project';
%     value1 = 'general';
% 
%     field2 = 'featureNames';
%     
%     field3 = 'featureAppearances';
%     field4 = 'objectiveName';
%     field5 = 'THzSource';
%     value2 = 

AstraFilesFolder = [pwd,'\AstraFiles\'];
astra_file_original = [AstraFilesFolder, 'rfgun.in'];

params = {'CaviP','CaviA','SoleB','CaviP','CaviA','QuadK','QuadK','QuadK','DipoB'};
appearances = [1,1,1,2,2,1,2,3,1];
output_para = {'RMSBunchLength'};
THz_source = 'CSR';

% try
%     w=getCurrentWorker;
%     if isempty(w)
%         tempName = 'client';
%     else
%         tempName = num2str(w.ProcessId);
%     end
% catch
%     tempName = 'client';
% end
% projectName = tempname;

TempFolder = tempname;
[tempDir,newFolder]=fileparts(TempFolder());
projectName = num2str(randi(10^9-1,1));
TempFolder = [TempFolder,'\'];

% newFolder = ['myfolder',projectName];
% TempFolder = fullfile(tempdir,newFolder);
% TempFolder = [TempFolder,'\'];
mkdir(tempDir,newFolder);
% disp(TempFolder)

astra_file_modified = [TempFolder,projectName,'.in'];
out_dist_file = [TempFolder, projectName,'.1475.001'];
out_LandF_file = [TempFolder, projectName,'.LandF.001'];

supportFiles{1}=[AstraFilesFolder,'10000p_1pC.ini'];
supportFiles{2}=[AstraFilesFolder,'bfld_scand.dat'];
supportFiles{3}=[AstraFilesFolder,'CTF3_Ez_ASTRA.dat'];
supportFiles{4}=[AstraFilesFolder,'TWS_PSI_Sband_ASTRA.dat'];
copy_supportFiles_for_parallel(TempFolder, supportFiles);

idx = strcmp(output_para,'THzPeak');
if all(idx) % contains and only contain THzPeak
    output_para_temp = {'RMSBunchLength'};
    [~, data, bunchValid]= run_and_get_beam_para(astra_file_original, astra_file_modified, params, x, appearances,out_dist_file, out_LandF_file, output_para_temp);
    if bunchValid
        fitness = run_and_get_THz_para(data,THz_source, x);
    else
        fitness = NaN;
    end
elseif any(idx) % contains THzPeak and something else
    output_para_temp = output_para(~idx);
    [fitness, data, bunchValid]= run_and_get_beam_para(astra_file_original, astra_file_modified, params, x, appearances,out_dist_file, out_LandF_file, output_para_temp);
    if bunchValid
        fitness(end+1) = run_and_get_THz_para(data,THz_source, x);
    else
        fitness(end+1) = NaN;
    end
else
    fitness= run_and_get_beam_para(astra_file_original, astra_file_modified, params, x, appearances,out_dist_file, out_LandF_file, output_para); 
end


rmdir(TempFolder, 's');