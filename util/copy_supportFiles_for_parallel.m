function copy_supportFiles_for_parallel(destination_folder, files)

for ii=1:length(files)
    copyfile(files{ii}, destination_folder);
end