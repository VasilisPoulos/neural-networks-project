# **Neural Networks Project**

Exercise for the elective course Neural Networks, CSE UOI.

Πούλος Βασίλης, 2805\
Δημητρόπουλος Δημήτρης, 4352\
Πούλος Γρηγόρης, 4480

## **Δημιουργία Συνόλων Δεδομένων**

Για την δημιουργία των συνόλων δεδομένων έχει δημιουργηθεί το αρχείο 
`generate_dataset.c` το οποίο με την εκτέλεση του δημιουργεί τα `training_set.txt`
και `test_set.txt` για την εκπαίδευση και τον έλεγχο του νευρωνικoύ δικτύου της 
πρώτης άσκησης και το `dataset2.txt` για τον αλγόριθμο kmeans της δεύτερης άσκησης.

Για την γραφική αναπαράσταση των δεδομένων χρησιμοποιείται το `plot_dataset.py` το
οποίο δέχεται είτε ένα είτε δύο ορίσματα. Στην πρώτη περίπτωση το script δημιουργεί γραφική
αναπαράσταση δεδομένων από

```bash
$
```

## **Άσκηση 1**

### **Παράμετροι και Εκτέλεση**

Στην άσκηση αυτή υλοποιήθηκε το MLP Νευρωνικό Δίκτυο το οποίο μπορεί να 
παραμετροποιηθεί και να εκτελεστεί με αρχείο `runMlp.java`. 

+ Η παραμετροποίηση πραγματοποιείται δίνοντας τιμές στις μεταβλητές στο αρχείο όπως 
φαίνεται παρακάτω: 
```java
        int numOfHiddenLayers = 3; // type "2" or "3"
        int D = 2;
        int H1 = 10;
        int H2 = 8;
        int H3 = 8; // Ignored if numOfHiddenLayers == 2
        int K = 4;
        String hiddenLayerActivationFunction = "tanh"; //type "relu" or "tanh"
        double LEARNING_RATE = 0.0009;
        int BATCH_SIZE = 1;
        int MINIMUM_EPOCHS = 700;
        double TERMINATION_THRESHOLD = 0.1;
```

+ Για την εκτέλεση: 
```bash
$ javac Neuron.java Mlp.java runMlp.java
$ java runMlp 
```
Κατά την εκτέλεση του προγράμματος καλούνται τρεις βασικές συναρτήσεις: 
```java
        mlp.initWeights();
        mlp.gradientDescent("../../data/training_set.txt");
        mlp.testNetwork("../../data/test_set.txt");
``` 
+ Με την `initWeights()` αρχικοποιούνται όλα τα βάρη και οι πολώσεις του δικτύου 
σε τυχαίους αριθμούς μεταξύ -1 και 1.
+ Με την `gradientDescent()` φορτώνεται το αρχείο που περιέχει τα δεδομένα εκπαίδευσης 
του δικτύου και τρέχει ο αλγόριθμος για τις παραμέτρους που έχουν οριστεί. Για την 
λειτουργία του αλγορίθμου έχουν υλοποιηθεί οι συναρτήσεις `forwardPass(double[] networkInput)`,
που δέχεται ως όρισμα μια είσοδο για το δίκτυο και επιστρέφει την έξοδο του και η 
`backprop(double[] networkInput, double[] data_label)` η οποία δέχεται ως όρισμα μια είσοδο 
και την επιθυμητή κατηγορία που θα πρέπει να επιστρέψει το δίκτυο και υπολογίζει το σφάλμα και
την μερική παράγωγο σε κάθε νευρώνα. Ανάλογα με τον αριθμό τον mini-batches που έχουμε δώσει 
καλείται η συνάρτηση `updateWeights()` η οποία ενημερώνει τα βάρη και τις πολώσεις στο δίκτυο 
χρησιμοποιώντας τον ρυθμό μάθησης που έχει οριστεί. Η εκπαίδευση τρέχει για το ελάχιστο των 700 
εποχών (MINIMUM_EPOCHS) όπως ορίζεται και συνεχίζει να τρέχει μέχρι η διαφορά δύο διαδοχικών 
σφαλμάτων να είναι μικρότερη απο το κατώφλι που ορίζεται στην αρχή του προγράμματος 
(TERMINATION_THRESHOLD). Καθ' όλη την διάρκεια της εκτέλεσης τυπώνεται το συνολικό σφάλμα 
εκπαίδευσης σε κάθε εποχή και κατά τον τερματισμό γράφονται όλα τα αποτελέσματα στο αρχείο 
`mlp_output.txt`.
+ Με την `testNetwork()` φορτώνονται τα δεδομένα ελέγχου του δικτύου και ύστερα δίνονται ως είσοδο
συγκρίνοντας την έξοδο του με την επιθυμητή. Στο τέλος της εκτέλεσης της τυπώνεται το ποσοστό
των σωστών αποφάσεων στο σύνολο ελέγχου και γράφονται στο αρχείο `mlp_error.txt` τα παραδείγματα 
ελέγχου με το σύμβολο "+" εάν το δίκτυο επέλεξε την σωστή κατηγορία για το συγκεκριμένο παράδειγμα
και "-" σε αντίθετη περίπτωση.

### How to run

### **Συνάρτηση Εξόδου**

Για την συνάρτηση εξόδου του δικτύου χρησιμοποιείται η λογιστική συνάρτηση (sigmoid) 
καθώς αυτό χρησιμοποιείται για την ταξινόμηση δεδομένων σε κατηγορίες.  

Παρακάτω δίνονται οι παράμετροι που χρησιμοποιήθηκαν για το δίκτυο με το 
μικρότερο σφάλμα γενίκευσης: 
(μπορει να αλλαξει αυτο)

### **Παρατηρήσεις**




## **Άσκηση 2** 
## How to run
Για να εκτελεστεί ο kmeans και να εμφανιστούν τα
αποτελέσματα (plot) χρησιμοποιείται το bash script
`plot_kmeans.sh`
ως:

```bash
$bash plot_kmeans.s
```

To script θα κάνει compile τα απαραίτητα αρχεία για
την παραγωγή του εκτελέσιμου, θα εκτελέσει τον kmeans 20 φορές με τυχαία
επιλεγμένα αρχικά κέντρα, θα κρατήσει την λύση με το μικρότερο σφάλμα
ομαδοποίησης και στην συνέχεια θα καλέσει το `plot_dataset.py` για να
εμφανίσει το αποτέλεσμα.

Η παραπάνω εκτέλεση χρησιμοποιεί τη μεταβλητή `NUM_OF_CLUSTERS` που
ορίζει το επιθυμητο αριθμό κέντρων, η οποία έχει οριστεί με define στο
αρχείο `kmeans.h` και μπορεί να αλλαχθεί σε οποιονδήποτε αριθμό.

Με τον παραπάνω τρόπο εκτελείται ο kmeans για Μ = 3, 5, 7, 9, 11, 13 κέντρα
και επιστρέφονται τα παρακάτω αποτελέσματα.

|                                          |                                          |
| :--------------------------------------: | :--------------------------------------: |
|  ![kmeans3](images/kmeans_3.png) M = 3   |  ![kmeans5](images/kmeans_5.png) M = 5   |
|  ![kmeans7](images/kmeans_7.png) M = 7   |  ![kmeans9](images/kmeans_9.png) M = 9   |
| ![kmeans11](images/kmeans_11.png) M = 11 | ![kmeans13](images/kmeans_13.png) M = 13 |

Το καλύτερο σφάλμα ανα περίπτωση τυπώνεται στο τερματικό. Οι τιμές
χρησιμοποιούνται για τον σχεδιασμό του γραφήματος που δείχνει την μεταβολή
του σφάλματος ομαδοποίησης με τον αριθμό ομάδων.

![min-values](images/plot_min_values.png)

![elbow-method](images/chart.png)

Στο γράφημα μπορεί να παρατηρηθεί ότι ενώ αρχικά η μεταβολή του σφάλματος είναι
μεγάλη, σταδιακά μικραίνει. Μετά τα 9 κεντρα η μεταβολή του σφάλματος είναι
πλέον τόσο μικρή που, σε συνδυασμό με το αυξημένο κόστος, δεν
δικαιολογεί την επιλογή περισσότερων κέντρων καθώς δεν θα αντιπροσωπεύουν
με πολυ καλύτερο τρόπο τα δεδομένα. Αυτη η ευρετική μέθοδος ονομάζεται elbow η
knee method. Είναι γνωστή στην ανάλυση συστάδων (clustering analysis) και βοηθάει
στον εμπειρικό προσδιορισμό του πραγματικό αριθμού ομάδων σε ένα σύνολο
δεδομένων.
