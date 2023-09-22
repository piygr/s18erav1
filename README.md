# Session 18 - Assignment

## Part 1 - UNet Implementation from scratch
Need to train it 4 times:

- MP+Tr+BCE
- MP+Tr+Dice Loss
- StrConv+Tr+BCE
- StrConv+Ups+Dice Loss

### s18part1-UNet.ipynb
Notebook for the UNet implementation

**MP+Tr+BCE**
```
Epoch  0
Train Loss:  0.918443500995636
Val Loss:  0.9185645580291748
Epoch  1
Train Loss:  0.80387282371521
Val Loss:  0.8910126686096191
Epoch  2
Train Loss:  0.833260715007782
Val Loss:  0.8598655462265015
Epoch  3
Train Loss:  0.851458728313446
Val Loss:  1.2130366563796997
Epoch  4
Train Loss:  0.8389899730682373
Val Loss:  1.084434151649475
Epoch  5
Train Loss:  0.8282903432846069
Val Loss:  0.9525937438011169
Epoch  6
Train Loss:  0.811596155166626
Val Loss:  0.8675016164779663
Epoch  7
Train Loss:  0.7977421879768372
Val Loss:  0.8013290166854858
Epoch  8
Train Loss:  0.7816001772880554
Val Loss:  0.7775134444236755
Epoch  9
Train Loss:  0.7644016146659851
Val Loss:  0.7651811242103577
```
<img width="555" alt="Screenshot 2023-09-23 at 1 38 15 AM" src="https://github.com/piygr/s18erav1/assets/135162847/daf28c26-fb87-4071-81d7-22b46fb7cff9">


**MP+Tr+Dice**
```
Epoch  0
Train Loss:  0.6144148707389832
Val Loss:  0.44544461369514465
Epoch  1
Train Loss:  0.3424493372440338
Val Loss:  0.4122726321220398
Epoch  2
Train Loss:  0.3177323043346405
Val Loss:  0.3573381006717682
Epoch  3
Train Loss:  0.33076608180999756
Val Loss:  0.5089702010154724
Epoch  4
Train Loss:  0.3256569504737854
Val Loss:  0.557777464389801
Epoch  5
Train Loss:  0.3112836480140686
Val Loss:  0.557707667350769
Epoch  6
Train Loss:  0.2867153286933899
Val Loss:  0.35730454325675964
Epoch  7
Train Loss:  0.2597164809703827
Val Loss:  0.27520614862442017
Epoch  8
Train Loss:  0.22615401446819305
Val Loss:  0.25413215160369873
Epoch  9
Train Loss:  0.19882728159427643
Val Loss:  0.18406975269317627
```
<img width="562" alt="Screenshot 2023-09-23 at 1 38 35 AM" src="https://github.com/piygr/s18erav1/assets/135162847/439735de-55d0-449e-904f-e4f4d4b9e8b9">


**StrConv+Tr+BCE**
```
Epoch  0
Train Loss:  1.0316799879074097
Val Loss:  0.928607702255249
Epoch  1
Train Loss:  0.8268437385559082
Val Loss:  0.7955427765846252
Epoch  2
Train Loss:  0.7828112840652466
Val Loss:  0.8132895827293396
Epoch  3
Train Loss:  0.7773500680923462
Val Loss:  0.8747822046279907
Epoch  4
Train Loss:  0.7850151658058167
Val Loss:  0.9614699482917786
Epoch  5
Train Loss:  0.7734285593032837
Val Loss:  0.9135054349899292
Epoch  6
Train Loss:  0.75990229845047
Val Loss:  0.8352715373039246
Epoch  7
Train Loss:  0.7512854337692261
Val Loss:  0.8207166790962219
Epoch  8
Train Loss:  0.7452548742294312
Val Loss:  0.9438756108283997
Epoch  9
Train Loss:  0.7344192862510681
Val Loss:  0.7702651023864746
Epoch  10
Train Loss:  0.720599889755249
Val Loss:  0.7678807377815247
Epoch  11
Train Loss:  0.7109583616256714
Val Loss:  0.7552934885025024
Epoch  12
Train Loss:  0.6984559893608093
Val Loss:  0.7087023854255676
Epoch  13
Train Loss:  0.6861688494682312
Val Loss:  0.685983419418335
Epoch  14
Train Loss:  0.6787976622581482
Val Loss:  0.6796802878379822
```
<img width="599" alt="Screenshot 2023-09-23 at 1 39 24 AM" src="https://github.com/piygr/s18erav1/assets/135162847/2de4b00b-58f9-4b87-9869-48901642c34d">


**StrConv+Ups+Dice**
```
Epoch  0
Train Loss:  0.5796436667442322
Val Loss:  0.45036059617996216
Epoch  1
Train Loss:  0.3332398235797882
Val Loss:  0.37693217396736145
Epoch  2
Train Loss:  0.29318302869796753
Val Loss:  0.327883243560791
Epoch  3
Train Loss:  0.2997015714645386
Val Loss:  0.4716005325317383
Epoch  4
Train Loss:  0.2909371554851532
Val Loss:  0.2919904589653015
Epoch  5
Train Loss:  0.29173535108566284
Val Loss:  0.5064186453819275
Epoch  6
Train Loss:  0.2863629460334778
Val Loss:  0.3359015882015228
Epoch  7
Train Loss:  0.2722666561603546
Val Loss:  0.3653419613838196
Epoch  8
Train Loss:  0.2581278085708618
Val Loss:  0.3702472150325775
Epoch  9
Train Loss:  0.24356268346309662
Val Loss:  0.35253384709358215
Epoch  10
Train Loss:  0.2351396083831787
Val Loss:  0.2311789095401764
Epoch  11
Train Loss:  0.21871574223041534
Val Loss:  0.21491171419620514
Epoch  12
Train Loss:  0.20134201645851135
Val Loss:  0.20177608728408813
Epoch  13
Train Loss:  0.18680065870285034
Val Loss:  0.17598333954811096
Epoch  14
Train Loss:  0.17647115886211395
Val Loss:  0.17214788496494293
```
<img width="581" alt="Screenshot 2023-09-23 at 1 39 34 AM" src="https://github.com/piygr/s18erav1/assets/135162847/738c1ac3-44a7-4a08-a857-bac216bfe780">


## Part 2 - VAE Implementation with Label as Input Embedding
### Strategy
- Created label-embedding on the lines of CLIP and passed as input along with the image.
- The label-embedding is multiplied to the encoder output of the image eg.
```
x_encoded = self.encoder(x)
x_encoded_embedded = x_encoded * self.label_embed(label)
```

### MNIST output with wrong labels
<img width="598" alt="Screenshot 2023-09-23 at 1 45 54 AM" src="https://github.com/piygr/s18erav1/assets/135162847/8f2c711d-e223-4f89-8caf-31527f8c521c">
<img width="597" alt="Screenshot 2023-09-23 at 1 46 05 AM" src="https://github.com/piygr/s18erav1/assets/135162847/d73d3abd-4aef-4e2f-ba3d-b47755d6fef1">


### CIFAR10 output with wrong labels
<img width="594" alt="Screenshot 2023-09-23 at 1 52 41 AM" src="https://github.com/piygr/s18erav1/assets/135162847/62095bd9-b0d0-4da7-b8c9-7f581bce45b3">
<img width="585" alt="Screenshot 2023-09-23 at 1 52 56 AM" src="https://github.com/piygr/s18erav1/assets/135162847/ddeeeec5-7bf3-4115-9f28-875828caf258">



