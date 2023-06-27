dataset = "E:\Marcin\Magisterka\Dane\Reco\MFN";
dataset_name = "mfn";
dataset_lower = lower(dataset_name);
library = "facenet";

path_file = dataset + "\" + dataset_lower + "_reco_" + library + ".csv";

recos = readtable(path_file, 'ReadVariableNames', false, "Delimiter",',');
good = 0;
bad = 0;

for i=1:1:height(recos)
    reco = recos(i,:);
    name1 = string(reco.Var1);
    name2_all = split(string(reco.Var2),"'");
    name2 = name2_all(2);

    if strcmp(name1, name2)
        good = good+1;
    else
        bad = bad+1;
    end
end

good
bad
result = good/(good+bad)*100