% D: dimension of the hyper-vector being used
% N: size of the N-gram 
% sentimentAM: contains hyper-vectors associated to each sentiment
% iM: contains hyper-vectors associated to each character 

function message = sam
  assignin('base','lookupItemMemeory',@lookupItemMemeory);
  assignin('base','genRandomHV',@genRandomHV);
  assignin('base','cosAngle',@cosAngle);
  assignin('base','computeSumHV', @computeSumHV);
  assignin('base','buildSentimentHV', @buildSentimentHV);
  assignin('base','binarizeHV', @binarizeHV);
  assignin('base','binarizeSentimentHV', @binarizeSentimentHV);
  assignin('base','test', @test); 
  message='Done importing functions to workspace';
end

% function to generate a random hyper-vector with dimension D
function randomHV = genRandomHV(D)
    if mod(D,2) % dimension need to be even 
        disp ('Dimension is odd!!');
    else
        randomIndex = randperm (D); % produces a random array of integers from 1 to D
    	% next two lines randomly distributes 1 and
    	% -1 across the indexes of hyper-vector randomHV
        randomHV (randomIndex(1 : D/2)) = 1; 
        randomHV (randomIndex(D/2+1 : D)) = -1;
        % error can be generated at this point if the mean
        % of the values in hyper-vector is not close to 0
    end
end

% function used to provide 
function [itemMemory, randomHV] = lookupItemMemeory(itemMemory, key, D)
	% container itemMemory has data mapped for value key 
	% then assign the data to randomHV
    if itemMemory.isKey (key) 
        randomHV = itemMemory (key);
        %disp ('found key');
    else
    	% if the key is not found in container itemMemory
    	% then generate a Random hyper-vector and assign it to 
    	% the itemMemory(key) and assign same vector to randomHV
        itemMemory(key) = genRandomHV (D);
        randomHV = itemMemory (key);
    end
end

% function computes cos of the angle between two 
% hyper-vector
function cosAngle = cosAngle (u, v)
     cosAngle = dot(u,v)/(norm(u)*norm(v));
end


function [itemMemory, sumHV] = computeSumHV (buffer, itemMemory, N, D)
    %init
    block = zeros (N,D);
    sumHV = zeros (1,D);
    
    % iterating through every character in the buffer 
    for numItems =1:1:length(buffer)
        %read a key
        key = buffer(numItems);
        
        %shift read vectors
        block = circshift (block, [1,1]);
        [itemMemory, block(1,:)] = lookupItemMemeory (itemMemory, key, D); 

        %
        if numItems >= N
            nGrams = block(1,:);
            for i = 2:1:N
                nGrams = nGrams .* block(i,:); %element-wise multiplication
            end
            sumHV = sumHV + nGrams;
        end
    end
    
end

% function used in accuracy testing
% function binerizes a hyper-vector based on a certain threshold
function v = binarizeHV (v)
	threshold = 0;
	for i = 1 : 1 : length (v)
		if v (i) > threshold
			v (i) = 1;
		else
			v (i) = -1;
		end
	end
end

% function used in accuracy testing 
% function binarizes hyper-vectors of each sentiment 
function sentimentAM = binarizeSentimentHV (sentimentAM) 
    sentimentLabels = {'pos', 'neg'};
    
    for j = 1 : 1 : length (sentimentLabels)
        v = sentimentAM (char(sentimentLabels (j)));
		sentimentAM (char(sentimentLabels (j))) = binarizeHV (v);
    end      
	
end

function [iM, sentimentAM] = buildSentimentHV (N, D) 
    % initializing iM and sentimentAM as containers
    iM = containers.Map;
    sentimentAM = containers.Map;
    sentimentLabels = {'pos', 'neg'};
    sentimentAM('pos') = zeros(1,D);
    sentimentAM('neg') = zeros(1,D);
    
    % iterating through list of all the sentiments 
    for i = 1:1:length(sentimentLabels)
        fileList = dir (char(strcat('./aclImdb/train/',sentimentLabels(i),'/*.txt')));
        for j = 1:1:length(fileList)
            % code block to get the training data 
            % get the address that conatains training text for a particular sentiment 
            fileAddress = strcat('./aclImdb/train/',sentimentLabels(i),'/',fileList(j).name);
            % opens the training file 
            fileID = fopen (char(fileAddress), 'r');
            % reads the text in the training file into the buffer
            buffer = fscanf (fileID,'%c');
            % closes the training filr 
            fclose (fileID);
            % print message that the file is loaded 
            fprintf('Loaded traning sentiment file %s\n',char(fileAddress)); 
        
            % this computes the hyper-vector for a particular text 
            [iM, sentimentHV] = computeSumHV (buffer, iM, N, D);

            % store the trained language hyper-vector to sentimentAM
            sentimentAM (char(sentimentLabels (i))) = sentimentAM (char(sentimentLabels (i))) + sentimentHV;
        end
    end        
end

% code runs the test data on trained model to evaluate the accuracy 
function accuracy = test (iM, sentimentAM, N, D)
	total = 0;
	correct = 0;
    sentimentLabels = {'pos', 'neg'};
    for j=1:1:length(sentimentLabels)
        fileList = dir (char(strcat('./aclImdb/test/',sentimentLabels(j),'/*.txt')));
        for i=1:1:length(fileList)
            actualLabel = char (sentimentLabels(j)); 
       
            fileAddress = strcat('./aclImdb/test/',sentimentLabels(j),'/', fileList(i).name);
            fileID = fopen (char(fileAddress), 'r');
            buffer = fscanf (fileID, '%c');
            fclose (fileID);
            fprintf ('Loaded testing text file %s\n', char(fileAddress)); 
        
            [iMn, textHV] = computeSumHV (buffer, iM, N, D);
            %textHV = binarizeHV (textHV);
            if iM ~= iMn
                fprintf ('\n>>>>>   NEW UNSEEN ITEM IN TEST FILE   <<<<\n');
                exit;
            else
                maxAngle = -1;
                for l = 1:1:length(sentimentLabels)
                    angle = cosAngle(sentimentAM (char(sentimentLabels (l))), textHV);
                    if (angle > maxAngle)
                        maxAngle = angle;
                        predictSentiment = char (sentimentLabels (l));
                    end
                end
                if predictSentiment == actualLabel
                    correct = correct + 1;
                else
                    fprintf ('%s --> %s\n', actualLabel, predictSentiment);
                end
                total = total + 1;
            end
            accuracy = correct / total;
        end
    end
end





