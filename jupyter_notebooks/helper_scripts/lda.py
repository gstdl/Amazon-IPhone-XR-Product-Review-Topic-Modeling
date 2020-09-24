from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class LDA:
    def __init__(self, texts):
        self.texts = texts
        self.docs = [data.split() for data in texts]
        self.dictionary = corpora.Dictionary(self.docs)
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.docs]

    def __iter__(self):
        self.length = len(self.texts)
        self.n = 0
        return self

    def __next__(self):
        token = self.bow_corpus[self.n]
        self.n += 1
        if self.n == self.length:
            self.n == 1
        return [(self.dictionary[ii], freq) for ii, freq in token]

    def filter_extremes(self, keep_n, no_below, no_above):
        self.dictionary.filter_extremes(
            keep_n=keep_n, no_below=no_below, no_above=no_above
        )
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.docs]

    def fit(
        self,
        num_topics,
        alpha="symmetric",
        beta=None,
        passes=2,
        random_state=9,
        tuning=False,
    ):
        self.model = models.ldamodel.LdaModel(
            self.bow_corpus,
            num_topics=num_topics,
            alpha=alpha,
            eta=beta,
            id2word=self.dictionary,
            passes=passes,
            random_state=random_state,
        )
        # calculate perplexity score (the lower the better)
        self.perplexity_score_ = self.model.log_perplexity(self.bow_corpus)
        # calculate coherence score (the higher the better)
        self.coherence_model = models.CoherenceModel(
            model=self.model,
            texts=self.docs,
            dictionary=self.dictionary,
            coherence="c_v",
        )
        self.coherence_score_ = self.coherence_model.get_coherence()
        self.coherence_score_per_topic_ = self.coherence_model.get_coherence_per_topic()
        if not tuning:
            pyLDAvis.enable_notebook()
            vis = pyLDAvis.gensim.prepare(self.model, self.bow_corpus, self.dictionary)
            self.visualize_topics_ = pyLDAvis.display(vis)
        else:
            self.visualize_topics_ = 'Set tuning parameter in fit function to "False" to visualize LDA result!'
            return self.coherence_score_

    def print_output(self):
        for idx, topic in self.model.print_topics(-1):
            print(
                "Topic: {}\tCoherence Score: {:.4f}\nWords: {}".format(
                    idx, self.coherence_score_per_topic_[idx], topic
                ),
                end="\n\n########\n\n",
            )
        else:
            print(
                "Perplexity Score: {:.4f} \nOverall Coherence Score: {:.4f}".format(
                    self.perplexity_score_, self.coherence_score_
                )
            )

    def find_best_num_topics(self, num_topic_range=np.arange(2, 61, 1)):
        self.tune(num_topic_range=num_topic_range, alpha_range=['symmetric'], beta_range=['symmetric'])
        data = [(i['num_topics'], i['coherence_score']) for i in self.tuning_results_]
        x, y = list(), list()
        for k, cs in data:
            x.append(k)
            y.append(cs)
        fig, ax = plt.subplots(figsize = (18, 6))
        _ = ax.plot(x,y)
        _ = ax.set_title('Optimal Number of Topics Grid Search')
        _ = ax.set_ylabel('Coherence Score')
        _ = ax.set_xlabel('Number of Topics')
        fig.show()
        
    # Default set of hyperparameter values for hyperparameter tuning
    num_topic_range = np.arange(2, 101, 10)
    alpha_range = list(np.arange(0.0, 1.1, 0.25))
    alpha_range.append("symmetric")
    alpha_range.append("asymmetric")
    beta_range = list(np.arange(0.0, 1.1, 0.25))
    beta_range.append("symmetric")

    def tune(
        self,
        random_grid_search=False,
        n=10,
        num_topic_range=num_topic_range,
        alpha_range=alpha_range,
        beta_range=beta_range,
    ):
        self.tuning_results_ = list()
        for n_topic in num_topic_range:
            for alpha in alpha_range:
                for beta in beta_range:
                    self.tuning_results_.append(
                        {"num_topics": n_topic, "alpha": alpha, "beta": beta}
                    )
        if random_grid_search:
            self.tuning_results_ = np.random.choice(self.tuning_results_, n, False)

        self.best_coherence_score_ = -np.inf
        idx_best_coherence_score = 0
        progress_bar = tqdm(range(len(self.tuning_results_)), desc="tuning")
        for i in progress_bar:
            coherence_score = self.fit(**self.tuning_results_[i], tuning=True)
            if coherence_score > self.best_coherence_score_:
                self.best_coherence_score_ = coherence_score
                self.best_params_ = self.tuning_results_[i].copy()
                idx_best_coherence_score = i
            self.tuning_results_[i]["coherence_score"] = coherence_score
            progress_bar.set_postfix(
                {
                    "max_coherence_score": self.best_coherence_score_,
                    "best_params": str(self.best_params_),
                }
            )
        else:
            self.fit(**self.best_params_)
            print("Finished hyperparameter tuning, model updated!")
