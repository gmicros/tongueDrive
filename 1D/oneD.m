clear; clc; close all

load('TDS_Raw_Sensors_Data.mat');

fts = size(sensorTestRaw(1).data, 2);

%% figure out how much stuff is in here
for i = 1:size(sensorTrainRaw, 2)
    obs(i) = size(sensorTrainRaw(i).data, 1);
end

input = zeros(sum(obs), fts);
target = zeros(sum(obs),1);
ind = 1;
%% put in a reasonable array
for i = 1:size(sensorTestRaw, 2)    
    input(ind : ind + obs(i) - 1, :) = sensorTrainRaw(i).data;
    target(ind : ind + obs(i) - 1) = repmat(sensorTrainRaw(i).target, obs(i), 1);
    ind = ind + obs(i);
end

% sort this stuff so that it look legit
[target, i] = sort(target);
input = input(i, :);

[Wkj, Wji] = twoLayerAnn(input', target', 10, 1000);

