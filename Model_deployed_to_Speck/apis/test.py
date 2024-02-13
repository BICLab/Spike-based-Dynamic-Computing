import torch
from tqdm import tqdm
from spikingjelly.clock_driven import functional
import sinabs.layers as sl

def reset_state(model):
    for lyrs in model.modules():
        if isinstance(lyrs, sl.IAF):
            lyrs.reset_states(randomize=True)

def test(args):
    args.test_loss = 0
    args.test_correct = 0
    bar_test = tqdm(total=len(args.test_loader))
    for _, (input, labels) in enumerate(args.test_loader):
        
        if args.spike_mode == 'IFCell':
            functional.reset_net(args.model)
        elif args.spike_mode == 'sinabs':
            reset_state(args.model)

        b = input.size()[0]

        input = input.reshape(
            b * args.clip,
            input.size()[2],
            input.size()[3],
            input.size()[4],
            input.size()[5]
        )
        input = input.float().to(args.device)
        if len(labels.shape) == 3:
            labels = labels.reshape(
                b * args.clip,
                labels.size()[2]
            )
            labels = labels.float().to(args.device)
        else:
            labels = labels.reshape(
                b * args.clip,
                labels.size()[2],
                labels.size()[3]
            )
            labels = labels[:, 1, :].float().to(args.device)

        outputs, _ = args.model(input)

        _, label = torch.max(labels.data, 1)

        loss = args.criterion(outputs, label)

        _, predicted = torch.max(outputs.data, 1)
        _, labelTest = torch.max(labels.data, 1)

        for i in range(b):
            predicted_clips = predicted[i * args.clip:(i + 1) * args.clip]
            labelTest_clips = labelTest[i * args.clip:(i + 1) * args.clip]
            test_clip_correct = (predicted_clips == labelTest_clips).sum().item()
            if test_clip_correct / args.clip > 0.5:
                args.test_correct += 1

        args.test_loss += loss.item() / args.clip


        bar_test.update()
        bar_test.set_description("Test:Epoch[%d/%d]" % (args.epoch + 1, args.num_epochs))
        bar_test.set_postfix(Loss=loss.item())

    bar_test.close()
