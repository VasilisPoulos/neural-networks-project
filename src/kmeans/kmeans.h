#define NUM_OF_CLUSTERS 10

typedef struct {
	double	x;
	double	y;
	int		group;
} Cluster;

typedef struct {
    double error;
    double x;
    double y;
} Kmeans;

float** read_file(char* filename);
float* parse_line(char* line);
int get_num_of_lines(FILE *fp);
void set_labels(float** dataset, int len_of_dataset);
void reset_array(float array[NUM_OF_CLUSTERS][3]);
void reposition_cluster_centers(float** dataset, \
    float cluster_sum_info[NUM_OF_CLUSTERS][3], \
    Cluster* cluster_list, int len_of_dataset);
void intialize_clusters(Cluster* cluster_list, float** dataset, \
    int len_of_dataset);
void print_tables(float cluster_sum_info[NUM_OF_CLUSTERS][3], Cluster* cluster_list, \
    int epoch);
int get_file_len(char* filename);
void write_labeled_dataset_to_file(char* filename, float** dataset, int len_dataset);
void write_kmeans_clusters_to_file(char* filename, Cluster* cluster_list);
int clusters_converged(Cluster* cluster_list, float previous_clusters[NUM_OF_CLUSTERS][2]);
float* error_calc(Cluster* cluster_list, float** dataset, int len_of_dataset);
Kmeans* kmeans(int num);
