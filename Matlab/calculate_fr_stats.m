dataset = "E:\Marcin\Magisterka\Dane\Reco\LFW";
dataset_name = "lfw";
dataset_lower = lower(dataset_name);
library = "facenet15";

path_file = dataset + "\" + dataset_lower + "_reco_" + library + ".csv";

recos = readtable(path_file, 'ReadVariableNames', false);
good = 0;
bad = 0;

for i=1:1:height(recos)
    reco = recos(i,:);
    name1_all = split(string(reco.Var1),"_");
    name1 = name1_all(1) + "_" + name1_all(2);
    name2_all1 = split(string(reco.Var2),"'");
    name2_all = split(name2_all1(2),"_");
    name2 = name2_all(1) + "_" + name2_all(2);

    if strcmp(name1, name2)
        good = good+1;
    else
        bad = bad+1;
    end
end

good
bad
result = good/(good+bad)*100
