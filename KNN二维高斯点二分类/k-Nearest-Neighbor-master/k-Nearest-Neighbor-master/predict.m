function test_pred = predict(y,idx)
average_distances=mean(y(idx)); %average of labels corresponding to indices that have smallest distances
if mean(average_distances)>=0.7
   test_pred=1;
else
   test_pred=0;
end
