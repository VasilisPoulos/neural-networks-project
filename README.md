# **Neural Networks Project**
Exercise for the elective course Neural Networks, CSE UOI.

Πούλος Βασίλης, 2805\
Δημητρόπουλος Δημήτρης, 4352\
Πούλος Γρηγόρης, 4480

## **Δημιουργία Συνόλων Δεδομένων** 

Για την δημιουργία των συνόλων δεδομένων έχει δημιουργηθεί το αρχείο 
`generate_dataset.c` το οποίο με την εκτέλεση του δημιουργεί τα `training_set.txt`
και `test_set.txt` για την εκπαίδευση και την δοκιμή του νευρωνικoύ δικτύου της 
πρώτης άσκησης και το `dataset2.txt` για τον αλγόριθμο kmeans της ο δεύτερης άσκησης.

Για την γραφική αναπαράσταση των δεδομένων χρησιμοποιείται το `plot_dataset.py` το 
οποίο δέχεται είτε ένα είτε δύο ορίσματα. Στην πρώτη περίπτωση το script δημιουργεί γραφική 
αναπαράσταση δεδομένων από 

```bash
$ 
```

## **Άσκηση 1**

Στην άσκηση αυτή υλοποιήθηκε το MLP Νευρωνικό Δίκτυο το οποίο μπορεί να 
παραμετροποιηθεί και να εκτελεστεί με αρχείο `runMlp.java`. 

Για την εκτέλεση: 
```bash
$ javac Neuron.java Mlp.java runMlp.java
$ java runMlp 
```
Παρακάτω δίνονται οι παράμετροι που χρησιμοποιήθηκαν για το δίκτυο με το 
μικρότερο σφάλμα γενίκευσης: 

(μπορει να αλλαξει αυτο)
```java
int numOfHiddenLayers = 3; // type "2" or "3"
        int D = 2;
        int H1 = 10;
        int H2 = 8;
        int H3 = 8; // Ignored if numOfHiddenLayers == 2
        int K = 4;
        double LEARNING_RATE = 0.0009;
        int BATCH_SIZE = 1;
        int MINIMUM_EPOCHS = 700;
        double TERMINATION_THRESHOLD = 0.001;
        String hiddenLayerActivationFunction = "tanh"; //type "relu" or "tanh"
```






## **Άσκηση 2** 
## How to run

```bash
$ make all
$ ./run
```
