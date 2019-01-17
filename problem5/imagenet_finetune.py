# from os.path import exists
# from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
# platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

# !pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl

# !pip3 install torchvision

# !pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
# !pip install Pillow==4.0.0
# !pip install PIL
# !pip install image


from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

NUM_EPOCH = 10


class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet50_CIFAR, self).__init__()
        # Initialize ResNet 50 with ImageNet weights
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        backbone = nn.Sequential(*modules)
        # Create new layers
        self.backbone = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, img):
        # Get the flattened vector from the backbone of resnet50
        out = self.backbone(img)
        # processing the vector with the added new layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)
 
def Softmax(x):
          result= np.zeros((x.shape[0], x.shape[1]))
          M,N = x.shape
          for m in range(M):
              Sum = np.sum(np.exp(x[m, :]))
              result[m, :] = np.exp(x[m, :])/Sum
          return result; 

def train():
  ## Define the training dataloader
  transform = transforms.Compose([transforms.Resize(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))])
  trainset = datasets.CIFAR10('./data', download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                        shuffle=True, num_workers=0)

  # Create model, objective function and optimizer
  model = ResNet50_CIFAR()
  model=model.cuda()
  criterion = nn.CrossEntropyLoss()
  criterion=criterion.cuda()
  optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),
                         lr=0.001, momentum=0.9)
    # Do the training
  for epoch in range(1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
      # get the inputs
      inputs = inputs.cuda()
      labels = labels.cuda()

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      if i % 20 == 19:    # print every 20 mini-batches
        print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 20))
        running_loss = 0.0
  print('Finished Training')
  GetHTMLtable(model)

def GetHTMLtable(model):
  model.eval()

  test_transform = transforms.Compose([transforms.Resize(224),
                                  transforms.ToTensor()])
  testset = datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=test_transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                           shuffle=True, num_workers=0)

  import base64
  from io import BytesIO

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  table_rows = """<html><table border="1"><tr><th>IMAGE</th><th>CLASSIFICATION_SCORE</th><th>CLASS</th><th>CLASSIFICATION</th></tr>"""
  def tensor_img_to_uri(img):
    img = img.cpu()
    img=transforms.ToPILImage()(img)
    buffer = BytesIO()
    img.save(buffer, "GIF")
    myimage = buffer.getvalue()
    return "data:image/jpeg;base64,{}".format(base64.b64encode(myimage).decode("utf-8"))



  for i, (inputs, ActualClass) in enumerate(testloader,0):
    inputs = inputs.cuda() 
    outputs=model(inputs)
    print(ActualClass)
    PredClass = Softmax(outputs.cpu().data.numpy())
    ClassificationScore= []
    PredictedClass = [] 
    RightWrong = list()
    for j in range(PredClass.shape[0]):
      ClassificationScore.append(max(PredClass[j, :]))
      PredictedClass.append(np.argmax(PredClass[j, :]))
    print(PredictedClass)

    for k in range( len(ActualClass)):
      if( PredictedClass[k] == ActualClass[k]):
        RightWrong.append( 'Classified')
      else:
         RightWrong.append( 'Miss-Classified')

    print(RightWrong)
    for scoran, img, classan, HaYaNa in zip(ClassificationScore, inputs, PredictedClass, RightWrong):
      img_uri=tensor_img_to_uri(img)
      table_rows += "<tr><td><img src='{0}' /></td><td>{1}</td><td>{2}</td><td>{3}</td></tr>".format(img_uri, scoran, classes[classan], HaYaNa)

    print("Finished batch #",i)
    if i == 2:
      break
  table_rows += "</table></html>"
  print("exporting final")
  print(table_rows)

# with open("drive/My Drive/table_rows.html", "w") as text_file:
#   text_file.write(table_rows)



if __name__ == '__main__':
    train()