require('nn')

local flags = require('flags')

local BoxCriterion, parent = torch.class('nn.BoxCriterion', 'nn.Criterion')

function BoxCriterion:__init(lambda)
    parent.__init(self)

    self.lambda = lambda
    self.classCriterion = nn.ClassNLLCriterion()
    self.posCriterion = nn.MSECriterion()
end

function BoxCriterion:updateOutput(input, target)
    local totalLoss = 0.0
    
    for i=1, flags.grid * flags.grid do
        local loss = self.classCriterion:forward(input[i], target[i])   
        totalLoss = totalLoss + loss
    end
    totalLoss = totalLoss * self.lambda
    
    --[[
    local loss = self.posCriterion:forward( input[ flags.grid * flags.grid + 1],
                                           target[ flags.grid * flags.grid + 1] )
    totalLoss = totalLoss + loss
    --]]
    
    self.output = totalLoss
    return self.output
end

function BoxCriterion:updateGradInput(input, target)
    self.gradInput = {}
    
    for i=1, flags.grid * flags.grid do
        self.gradInput[i] = self.classCriterion:backward(input[i], target[i]):clone()
        self.gradInput[i] = self.gradInput[i] * self.lambda
    end
    
    --[[
    local grad = self.posCriterion:backward( input[ flags.grid * flags.grid + 1],
                                            target[ flags.grid * flags.grid + 1] )
    
    self.gradInput[ flags.grid * flags.grid + 1 ] = grad
    --]]

    return self.gradInput
end



