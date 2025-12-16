#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct node {
    int key;
    int value;
    struct node* next;
} node_t;

typedef struct bucket {
    node_t* head;
    pthread_mutex_t lock;   /* fine-grained */
} bucket_t;

/* Opaque in benchmark.c via: typedef struct hashtable hashtable_t; */
typedef struct hashtable {
    bucket_t* buckets;
    size_t nbuckets;
    pthread_mutex_t global_lock; /* coarse-grained */
} hashtable_t;

static inline size_t hidx(hashtable_t* ht, int key) {
    return (size_t)((unsigned int)key % (unsigned int)ht->nbuckets);
}

hashtable_t* ht_create(size_t nbuckets) {
    if (nbuckets == 0) nbuckets = 1;

    hashtable_t* ht = (hashtable_t*)calloc(1, sizeof(hashtable_t));
    if (!ht) return NULL;

    ht->nbuckets = nbuckets;
    ht->buckets = (bucket_t*)calloc(nbuckets, sizeof(bucket_t));
    if (!ht->buckets) { free(ht); return NULL; }

    pthread_mutex_init(&ht->global_lock, NULL);
    for (size_t i = 0; i < nbuckets; i++) {
        pthread_mutex_init(&ht->buckets[i].lock, NULL);
    }
    return ht;
}

static void free_list(node_t* n) {
    while (n) {
        node_t* next = n->next;
        free(n);
        n = next;
    }
}

void ht_free(hashtable_t* ht) {
    if (!ht) return;
    for (size_t i = 0; i < ht->nbuckets; i++) {
        free_list(ht->buckets[i].head);
        pthread_mutex_destroy(&ht->buckets[i].lock);
    }
    pthread_mutex_destroy(&ht->global_lock);
    free(ht->buckets);
    free(ht);
}

/* ---------------- Coarse-grained ---------------- */

void ht_insert_coarse(hashtable_t* ht, int key, int value) {
    pthread_mutex_lock(&ht->global_lock);
    size_t i = hidx(ht, key);
    node_t* n = (node_t*)malloc(sizeof(node_t));
    n->key = key;
    n->value = value;
    n->next = ht->buckets[i].head;
    ht->buckets[i].head = n;
    pthread_mutex_unlock(&ht->global_lock);
}

int ht_find_coarse(hashtable_t* ht, int key, int* out) {
    int found = 0;
    pthread_mutex_lock(&ht->global_lock);
    size_t i = hidx(ht, key);
    for (node_t* n = ht->buckets[i].head; n; n = n->next) {
        if (n->key == key) {
            if (out) *out = n->value;
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&ht->global_lock);
    return found;
}

int ht_erase_coarse(hashtable_t* ht, int key) {
    int erased = 0;
    pthread_mutex_lock(&ht->global_lock);
    size_t i = hidx(ht, key);
    node_t* cur = ht->buckets[i].head;
    node_t* prev = NULL;

    while (cur) {
        if (cur->key == key) {
            if (prev) prev->next = cur->next;
            else ht->buckets[i].head = cur->next;
            free(cur);
            erased = 1;
            break;
        }
        prev = cur;
        cur = cur->next;
    }
    pthread_mutex_unlock(&ht->global_lock);
    return erased;
}

/* ---------------- Fine-grained ---------------- */

void ht_insert_fine(hashtable_t* ht, int key, int value) {
    size_t i = hidx(ht, key);
    pthread_mutex_lock(&ht->buckets[i].lock);
    node_t* n = (node_t*)malloc(sizeof(node_t));
    n->key = key;
    n->value = value;
    n->next = ht->buckets[i].head;
    ht->buckets[i].head = n;
    pthread_mutex_unlock(&ht->buckets[i].lock);
}

int ht_find_fine(hashtable_t* ht, int key, int* out) {
    int found = 0;
    size_t i = hidx(ht, key);
    pthread_mutex_lock(&ht->buckets[i].lock);
    for (node_t* n = ht->buckets[i].head; n; n = n->next) {
        if (n->key == key) {
            if (out) *out = n->value;
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&ht->buckets[i].lock);
    return found;
}

int ht_erase_fine(hashtable_t* ht, int key) {
    int erased = 0;
    size_t i = hidx(ht, key);
    pthread_mutex_lock(&ht->buckets[i].lock);
    node_t* cur = ht->buckets[i].head;
    node_t* prev = NULL;

    while (cur) {
        if (cur->key == key) {
            if (prev) prev->next = cur->next;
            else ht->buckets[i].head = cur->next;
            free(cur);
            erased = 1;
            break;
        }
        prev = cur;
        cur = cur->next;
    }
    pthread_mutex_unlock(&ht->buckets[i].lock);
    return erased;
}

