X = [0,0,1,1,0,1,0,0,1,1,0,1,0]
Y = [0,0,0,1,1,0,1,0,0,1,1,0,1];
numIn, numHid, numOut = 1, 3, 1;
theta1 = ( 1 * sqrt ( 6 / ( numIn + numHid) ) * randn( numIn + numHid + 1, numHid ) )
theta2 = ( 1 * sqrt ( 6 / ( numHid + numOut ) ) * randn( numHid + 1, numOut ) )
theta1_grad = zeros(numIn + numHid + 1, numHid)
theta2_grad = zeros(numHid + 1, numOut);
epochs = 60000
alpha = 0.1
epsilon = 0.3;
hid_last = zeros(numHid, 1)
last_change1 = zeros(numIn + numHid + 1, numHid)
last_change2 = zeros(numHid + 1, numOut)
m = size(X,1);
function sigmoid(z)
    return 1.0 ./ (1.0 + exp(-z))
end;
for i = 1:epochs
    #forward propagation
    s = rand(1:(m-1))
    for j = s:m #for every training element
        y = Y[j,:]
        context = hid_last
        x1 = X[j,:]
        a1 = [x1, context, 1] #add bias, context units to input layer; 3x1
        z2 = theta1' * a1
        a2 = [sigmoid(z2), 1] #output hidden layer;
        hid_last = a2[1:end-1,1]
        z3 = theta2' * a2
        a3 = sigmoid(z3)
        #skip first element
        if j != s
            #calculate delta errors
            d3 = (a3 - y);
            d2 = (theta2 * d3) .* (a2 .* (1 - a2))
            #accumulate gradients
            theta1_grad = theta1_grad + (d2[1:numHid, :] * a1')';
            theta2_grad = theta2_grad + (d3 * a2')'
        end
    end
    theta1_change = alpha * (1/m)*theta1_grad + epsilon * last_change1;
    theta2_change = alpha * (1/m)*theta2_grad + epsilon * last_change2;
    theta1 = theta1 - theta1_change;
    theta2 = theta2 - theta2_change;
    last_change1 = theta1_change;
    last_change2 = theta2_change;
    #reset gradients
    theta1_grad = zeros(numIn + numHid + 1, numHid);
    theta2_grad = zeros(numHid + 1, numOut);
end
results = Float64[]
Xt = [0,0,1,1,0,1,0]
for j = 1:size(Xt,1) #for every training element
    context = hid_last
    x1 = Xt[j,:]
    a1 = [x1, context, 1] #add bias, context units to input layer
    z2 = theta1' * a1
    a2 = [sigmoid(z2); 1] #output hidden layer
    hid_last = a2[1:end-1,1]
    z3 = theta2' * a2
    a3 = sigmoid(z3)
    push!(results, a3[1])
end
println(int(round(results)))