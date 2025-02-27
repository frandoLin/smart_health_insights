from sentence_transformers import SentenceTransformer
import numpy as np


text = ["""
OBJECTIVE        To evaluate the sedation and analgesia power and security of sufentanil in intensive care unit ( ICU ) , and to compare the effect with fentanyl .
METHODS A multicenter randomized controlled trial was conducted .
METHODS Critical adult patients in ICU from 11 hospitals in Henan Province from June 2011 to January 2012 who needed analgesia based sedation were enrolled .
METHODS These patients were randomly divided into two groups with 300 cases in each group using the envelope method according to the hospital number and time sequence number of inclusion .
METHODS Exclusion criteria included the time of analgesia duration < 48 hours and who were under continuous renal replacement therapy ( CRRT ) treatment during analgesia .
METHODS 544 cases were enrolled finally , and there were 282 cases in sufentanil group and 262 in fentanyl group .
METHODS Before using the drug , there was no statistically significant difference in age , body weight , acute physiology and chronic health evaluation II ( APACHEII ) score , Glasgow coma scale ( GCS ) between sufentanil group and fentanyl group , and were comparable .
METHODS The goal of analgesia was faces pain scale ( FPS ) 2 .
METHODS If the dosage of sufentanil and fentanyl exceeded the upper limited dose ( sufentanil 0.3 gkg ( -1 ) h ( -1 ) , fentanyl 2 gkg ( -1 ) h ( -1 ) ) but FPS could not meet ( still > 2 ) , and maintained the upper limited doses of sufentanil and fentanyl and added midazolam , and FPS2 or Ramsay 3 could meet the standard .
METHODS The analgesia duration of all cases was 48-168 hours .
METHODS Related data were collected for statistical analysis .
RESULTS ( 1 ) Compared with the data before the analgesia , the mean arterial pressure ( MAP ) of sufentanil analgesia after analgesia at different time points were significantly decreased ( F = 6.061 , P < 0.001 ) and closed to the normal level , FPS at different time point score were decreased significantly after analgesia ( F = 259.389 , P < 0.001 ) , and the changes in pulse oxygen saturation ( SpO ( 2 ) ) , respiratory rate and pulse were not found .
RESULTS ( 2 ) Compared with before the analgesia , the white blood cell count ( WBC ) , neutrophil percentage ( N ) , platelet count ( PLT ) , aspartate transaminase ( AST ) , creatinine ( Cr ) , arterial partial pressure of carbon dioxide ( PaCO ( 2 ) ) , blood lactic acid , blood sugar , C-reactive protein ( CRP ) were markedly reduced after sufentanil analgesia ( WBC : 10.8 4.2 10 ( 9 ) / L vs. 14.2 11.510 ( 9 ) / L , F = 49.879 , P < 0.001 ; N : 0.806 0.104 vs. 0.815 0.128 , F = 5.768 , P = 0.017 ; PLT : 160.4 77.0 10 ( 9 ) / L vs. 166.1 89.010 ( 9 ) / L , F = 6.568 , P = 0.011 ; AST : 61.3 10.1 U/L vs. 90.9 26.9 U/L , F = 6.706 , P = 0.010 ; Cr : 86.7 71.8 mol/L vs. 119.6 56.0 mol/L , F = 30.303 , P < 0.001 ; PaCO ( 2 ) : 39.4 7.2 mmHg vs. 41.7 22.6 mmHg , F = 4.389 , P = 0.037 ; blood lactic acid : 1.9 1.2 mmol/L vs. 2.7 2.5 mmol/L , F = 4.883 , P = 0.028 ; blood sugar : 8.0 5.4 mmol/L vs. 9.7 7.6 mmol/L , F = 9.724 , P = 0.002 ; CRP : 64.8 20.7 mg/L vs. 114.0 55.9 mg/L , F = 4.883 , P = 0.028 ) .
RESULTS But there were no statistically significant differences in red blood cell count ( RBC ) , prothrombin time ( PT ) , activated partial thromboplastin time ( APTT ) , fibrinogen ( FIB ) , thrombin time ( TT ) , alanine aminotransferase ( ALT ) , total bilirubin ( TBil ) , albumin ( ALB ) , total protein ( TP ) blood urea nitrogen ( BUN ) , and arterial partial pressure of oxygen ( PaO ( 2 ) ) before and after sufentanil analgesia ( all P > 0.05 ) .
RESULTS ( 3 ) There was no statistically significant difference in effectiveness of sufentanil and five times dose of fentanyl ( P > 0.05 ) .
RESULTS There was no statistically significant difference in the proportion of sedative drugs midazolam usage [ 18.4 % ( 52/282 ) vs. 24.8 % ( 65/262 ) , ( 2 ) = 1.151 , P = 0.283 ] and the rate of analgesia success [ 44.3 % ( 125/282 ) vs. 48.9 % ( 128/262 ) , ( 2 ) = 0.571 , P = 0.450 ] and analgesia success [ 16.3 % ( 46/282 ) vs. 15.3 % ( 40/262 ) , ( 2 ) = 0.066 , P = 0.798 ] between sufentanil and fentanyl group .
RESULTS ( 4 ) Comparison of adverse reactions : the incidence of hypotension in sufentanil group was significantly lower than that in fentanyl group [ 3.2 % ( 9/282 ) vs. 6.9 % ( 18/262 ) , ( 2 ) = 3.900 , P = 0.048 ] , and other common adverse reactions , such as respiratory depression/pause , nausea/vomiting and dizziness , pruritus , allergy , slow heart beat ( bradycardia ) and metabolic reactions had no statistically significant difference .
RESULTS Addiction or tetanus of skeletal muscles was not found in both groups .
CONCLUSIONS     Compared with fentanyl , the analgesia efficacy of sufentanil is stronger .
CONCLUSIONS     Sufentanil has less physiological interference and lower incidence of adverse reactions for ICU patients .
""",
"""
OBJECTIVE	Depressive disorders are one of the leading components of the global burden of disease with a prevalence of up to 14 % in the general population .
OBJECTIVE	Numerous studies have demonstrated that pharmacotherapy combined with non-pharmacological measures offer the best treatment approach .
OBJECTIVE	Psycho-education as an intervention has been studied mostly in disorders such as schizophrenia and dementia , less so in depressive disorders .
OBJECTIVE	The present study aimed to assess the impact of psycho-education of patients and their caregivers on the outcome of depression .
METHODS	A total of 80 eligible depressed subjects were recruited and randomised into 2 groups .
METHODS	The study group involved an eligible family member and all were offered individual structured psycho-educational modules .
METHODS	Another group ( controls ) received routine counselling .
METHODS	The subjects in both groups also received routine pharmacotherapy and counselling from the treating clinician and were assessed at baseline , 2 , 4 , 8 , and 12 weeks using the Hamilton Depression Rating Scale ( HDRS ) , Global Assessment of Functioning ( GAF ) , and Psychological General Well-Being Index ( PGWBI ) .
METHODS	Results from both groups were compared using statistical methods including Chi-square test , Fisher 's exact test , Student 's t test , Pearson 's correlation coefficient , as well as univariate and multiple regression analyses .
RESULTS	Baseline socio-demographic and assessment measures were similar in both groups .
RESULTS	The study group had consistent improvement in terms of outcome measures with HDRS , GAF , and PGWBI scores showing respective mean change of -15.08 , 22 , and 60 over 12 weeks .
RESULTS	The comparable respective changes in the controls were -8.77 , 18.1 , and 43.25 .
CONCLUSIONS	Structured psycho-education combined with pharmacotherapy is an effective intervention for people with depressive disorders .
CONCLUSIONS	Psycho-education optimises the pharmacological treatment of depression in terms of faster recovery , reduction in severity of depression , and improvement in subjective wellbeing and social functioning .
"""]

class DocumentChunker:
    """A class for chunking documents using various strategies."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 max_tokens: int = 256,
                 chunk_size: int = 256):
        """
        Initialize the DocumentChunker with specified models and parameters.
        
        Args:
            model_name: Name of the sentence transformer model
            tokenizer_name: Name of the tokenizer model
            max_tokens: Maximum tokens per chunk
        """
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.model = SentenceTransformer(model_name)
        self.sentence_tokenizer = self.model.tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.sentence_tokenizer.encode(text))


    # fixed_size_chunking function splits a document into fixed-size chunks
    def fixed_size_chunking(self, document: str) -> list[str]:
        """
        Split document into fixed-size chunks.
        Uses a simple word-based approach.
        Args:
            document: Text document to split
            chunk_size: Maximum number of words per chunk
        Returns:
            List of text chunks
        """

        sentences = document.split('. ')
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            # Add sentence and a period
            if len(current_chunk.split()) + len(sentence.split()) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks


    # recursive_chunking function splits a document into chunks recursively
    def recursive_chunking(self, document: str) -> list[str]:
        """
        Split document into chunks recursively.
        Uses a simple word-based approach.
        Args:
            document: Text document to split
            chunk_size: Maximum number of words per chunk
        Returns:
            List of text chunks
        """

        len_doc = len(document.split())

        if len_doc <= self.chunk_size:
            return [document]
        
        mid = len_doc // 2
        left_chunk = document[:mid]
        right_chunk = document[mid:]

        if len(left_chunk.split()) <= self.chunk_size or len(right_chunk.split()) <= self.chunk_size:
            return [left_chunk, right_chunk]
        
        return self.recursive_chunking(left_chunk, self.chunk_size) + self.recursive_chunking(right_chunk, self.chunk_size)


    # overlapping_chunking function splits a document into overlapping chunks, 
    # which can be useful for tasks like summarization because it allows the model to see more context
    def overlapping_chunking(self, document: str, overlap: int = 50) -> list[str]:
        """
        Split document into chunks with specified overlap between consecutive chunks.
        Uses a simple word-based approach.
        
        Args:
            document: Text document to split
            chunk_size: Maximum number of words per chunk
            overlap: Number of words to overlap between chunks
        
        Returns:
            List of text chunks with specified overlap
        """
            
        words = document.split()
        chunks = []
        start = 0
        
        # Ensure overlap is less than chunk_size to prevent infinite loops
        overlap = min(overlap, self.chunk_size - 1) if self.chunk_size > 0 else 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            # Move start position forward, accounting for overlap
            start += self.chunk_size - overlap
        
        return chunks


    # sliding window chunking function splits a document into chunks by sliding a window over the document
    def sliding_window_chunking(self, document: str, stride: int = 50) -> list[str]:
        """
        Split document into chunks using a sliding window approach.
        Uses a simple word-based approach.
        Args:
            document: Text document to split
            chunk_size: Maximum number of words per chunk
            stride: Number of words to move the window for the next chunk
        Returns:
            List of text chunks
        """

        words = document.split()
        chunks = []

        # Ensure stride is less than chunk_size to prevent infinite loops
        stride = min(stride, self.chunk_size - 1) if self.chunk_size > 0 else 0

        # Iterate over the document with a sliding window
        for i in range(0, len(words) - self.chunk_size + 1, stride):
            chunk = ' '.join(words[i:i+self.chunk_size])
            chunks.append(chunk)
        
        return chunks


    # semantic_chunking function splits a document into chunks based on semantic similarity
    def semantic_chunking(self, document: str) -> list[str]:
        """
        Split document into chunks based on semantic similarity between sentences.
        
        Args:
            document: Text document to split
            chunk_size: Maximum number of words per chunk (approximate)
            
        Returns:
            List of text chunks grouped by semantic similarity
        """
        # Split document into sentences
        sentences = [s.strip() for s in document.split('.') if s.strip()]
        if not sentences:
            return []
        
        # Add periods back to sentences
        sentences_with_periods = [s + "." for s in sentences]
        
        # Get embeddings for each sentence
        sentence_embeddings = self.model.encode(sentences_with_periods)
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_word_count = 0
        
        # Start with the first sentence
        current_chunk_sentences.append(sentences_with_periods[0])
        current_chunk_word_count = len(sentences_with_periods[0].split())
        current_embedding = sentence_embeddings[0].reshape(1, -1)
        
        for i in range(1, len(sentences_with_periods)):
            sentence = sentences_with_periods[i]
            sentence_word_count = len(sentence.split())
            sentence_embedding = sentence_embeddings[i].reshape(1, -1)
            
            # Check if adding this sentence would exceed the word limit
            if current_chunk_word_count + sentence_word_count > self.chunk_size:
                # Save current chunk
                chunks.append(" ".join(current_chunk_sentences))
                
                # Start new chunk with this sentence
                current_chunk_sentences = [sentence]
                current_chunk_word_count = sentence_word_count
                current_embedding = sentence_embedding
                continue
            
            # Check semantic similarity with current chunk
            similarity = np.dot(current_embedding, sentence_embedding.T) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(sentence_embedding)
            )
            
            # If similarity is high, add to current chunk
            if similarity > 0.5:  # Threshold can be adjusted
                current_chunk_sentences.append(sentence)
                current_chunk_word_count += sentence_word_count
                # Update current embedding as average
                current_embedding = np.mean(
                    np.vstack([current_embedding, sentence_embedding]), axis=0
                ).reshape(1, -1)
            else:
                # If not similar enough, start a new chunk
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_chunk_word_count = sentence_word_count
                current_embedding = sentence_embedding
        
        # Add the final chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
        
        return chunks


    def enforce_token_limit(self, chunks : list[str], show_chunks=False) -> list[str]:
        """
        Ensures that all chunks respect token limits by further splitting any chunks
        that exceed the maximum token count.
        
        Args:
            chunks: List of text chunks from any chunking method
            model_name: Name of the model to use for tokenization
            max_tokens: Maximum number of tokens allowed per chunk
            
        Returns:
            List of chunks, each within the token limit
        """
        # Use the same tokenizer as the embedding model
        compliant_chunks = []
        
        for chunk in chunks:
            # Count tokens in the chunk
            token_count = self.count_tokens(chunk)
        
            # If chunk is already within limits, keep it as is
            if token_count <= self.max_tokens:
                compliant_chunks.append(chunk)
                continue
            
            # If chunk exceeds limit, split it by sentences
            sentences = chunk.split('. ')
            current_chunk = ""
            current_tokens = 0
            
            for i, sentence in enumerate(sentences):
                # Add period back except for the last sentence
                if i < len(sentences) - 1:
                    sentence = sentence + "."
                    
                # Count tokens for this sentence
                sentence_tokens = self.count_tokens(sentence)
                
                # Check if adding this sentence would exceed the token limit
                if current_tokens + sentence_tokens <= self.max_tokens:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_tokens += sentence_tokens
                else:
                    # Store the current chunk and start a new one
                    if current_chunk:
                        compliant_chunks.append(current_chunk.strip())
                    
                    # If a single sentence exceeds token limit, split it by words
                    if sentence_tokens > self.max_tokens:
                        words = sentence.split()
                        current_chunk = ""
                        current_tokens = 0
                        
                        for word in words:
                            word_tokens = self.count_tokens(word)
                            if current_tokens + word_tokens <= self.max_tokens:
                                current_chunk += " " + word if current_chunk else word
                                current_tokens += word_tokens
                            else:
                                compliant_chunks.append(current_chunk.strip())
                                current_chunk = word
                                current_tokens = word_tokens
                    else:
                        current_chunk = sentence
                        current_tokens = sentence_tokens
            
            # Don't forget the last chunk
            if current_chunk:
                compliant_chunks.append(current_chunk.strip())

        if show_chunks:
            print(f"Final chunks after enforcing token limit:\n{compliant_chunks}\n")
            print(f'the number of chunks is {len(compliant_chunks)}\n')
        
        return compliant_chunks


    def chunk(self, document: str, method: str = "fixed_size", token_limit: bool = True, **kwargs) -> list[str]:
            """
            Main chunking method that delegates to the appropriate strategy.
            
            Args:
                document: Text to chunk
                method: Chunking method to use (fixed_size, recursive, overlap, 
                    sliding_window, semantic, dynamic)
                **kwargs: Additional arguments for specific chunking methods
            
            Returns:
                List of text chunks
            """
            method_map = {
                "fixed_size": self.fixed_size_chunking,
                "recursive": self.recursive_chunking,
                "overlap": self.overlapping_chunking,
                "sliding_window": self.sliding_window_chunking,
                "semantic": self.semantic_chunking,
            }
            
            if method not in method_map:
                raise ValueError(f"Unknown chunking method: {method}")
                
            chunking_function = method_map[method]
            chunks = chunking_function(document, **kwargs)
            
            if token_limit:
                return self.enforce_token_limit(chunks)
            else:
                return chunks


if __name__ == "__main__":
    # Test the functions

    chunker = DocumentChunker()
    chunking_methods = ["fixed_size", "recursive", "overlap", "sliding_window", "semantic"]
    token_limit = False

    for method in chunking_methods:
        print(f"Using {method} chunking method:\n")

        # Process each document
        for i, document in enumerate(text):
            print(f"Document:{i+1}\n")

            if method == "overlap":
                chunks = chunker.chunk(document, method=method, token_limit=token_limit, overlap=50)
            elif method == "sliding_window":
                chunks = chunker.chunk(document, method=method, token_limit=token_limit, stride=50)
            else:
                chunks = chunker.chunk(document, method=method, token_limit=token_limit)

            # Print the chunks
            for j, chunk in enumerate(chunks):
                print(f"Chunk {j+1}:\n{chunk}\n")
            


   