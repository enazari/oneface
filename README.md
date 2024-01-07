# One Face to Rule Them All: A Study of Adversarial Attacks on Face Verification Systems

## Abstract
This paper delves into the vulnerabilities of Face Verification (FV) systems, particularly under Adversarial Attacks which subtly manipulate input images to deceive these systems. We introduce the **DodgePersonation Attack**, a novel approach to generating face images that impersonate a set of given identities while avoiding identification as any other specified identities. Our approach, which we call "One Face to Rule Them All", implements this with state-of-the-art performance, significantly outperforming the existing Master Face Attack in coverage percentage.

## Phase 1: Embedding Space Search
In this phase, we aim to identify points in the embedding space that correspond to the Attack Faces. This involves mapping the Match Set and Dodge Set into the embedding space, creating clusters, and then using a Genetic Algorithm (GA) with a specially developed fitness function to find the optimal points.

### Key Steps:
1. **Mapping to Embedding Space:** Faces are passed through MTCNN, normalized, and fed into the FM function.
2. **Cluster Generation:** Create clusters of the Match Set using the K-Means algorithm.
3. **Genetic Algorithm:** Utilize the LM-MA-ES GA to identify points in the embedding space satisfying the impersonation and dodging requirements.

### Fitness Function:
- The **DodgePersonation Fitness function** is designed to optimize the impersonation and dodging objectives, with separate thresholds and weights allowing flexible adjustment for more favorable results.

## Phase 2: Attack Face Generation
Here, we generate the actual Attack Face images from the points obtained in Phase 1. Starting with any pre-selected Source Face, we iteratively modify it to decrease the loss value, ensuring the changed image is closely mapped to the target point in the embedding space.

### Process:
1. **Source Face Modification:** Modify the source face image to force it to match the obtained embedding point while keeping changes minimal.
2. **Gradient Descent:** Calculate the derivatives of the FM function with respect to the input image and alter the image to decrease the loss value.

### Algorithm:
The algorithm iteratively modifies the source face using the Adam optimization algorithm and a clipping function to ensure the changes remain within a specified range.

## Conclusion
Our approach presents an improvement over existing methods, offering a more powerful and flexible way to attack FV systems. The paper provides a comprehensive understanding and a new taxonomy of Adversarial Attacks against FV systems, setting the groundwork for further exploration and defense mechanisms.

For the complete study, detailed methodologies, and further discussions, please refer to the [preprint paper](Link-to-the-paper).


# One Face to Rule Them All: Extra experiments


## Source Face Modification Algorithm


## Algorithm: One Face to Rule Them All - Phase 2: Attack Face Generation

**Input:**
- `x`: Source Face
- `\bar{y}`: point in the embedding space returned in Phase 1
- `\epsilon`: maximum change allowed for each pixel of the normalized image
- `iterations`: number of iterations

**Output:**
- `x_{adv}`: attack image

**Procedure:**
1. **While** `x` != MTCNN(`x`):
    - `x` <- MTCNN(`x`)
2. `x_{adv}` <- `x`
3. **For** each `iter` in `iterations`:
    - `\bar{x}_{adv}` <- FM(`x_{adv}`)
    - `loss` <- Dist(`\bar{x}_{adv}`, `\bar{y}`)
    - `gradients` <- \frac{\partial loss}{\partial x_{adv}}
    - `x_{adv}` <- Adam(`x_{adv}`, `gradients`)
    - `x_{adv}` <- clip(`x_{adv}`, [`x_{adv}-\epsilon`, `x_{adv}+\epsilon`])
4. **Return** `x_{adv}`

**Caption:** One Face to Rule Them All Algorithm - Phase 2: Attack Face Generation.



## Taxonomy Discussion 

Taking a closer look at FV systems from the perspective of an attacker, we can explore the conditions under which the systems can be deceived. The objective of the attacker is to create a set of images that \emph{look like them} to the human eye (i.e., the Source Face is a face image of the attacker) but are identified as a different identity or identities by the FV system. Below, we present a list of real-world scenarios that correspond to specific examples of the cases in our proposed taxonomy, shown in Figure~\ref{fig:taxonomy}.

\textbf{Null Attack.} 
This scenario occurs when there is no intention to impersonate or dodge any identity. This is a trivial case where any valid input is a solution.

\textbf{Single Identity Dodging.} In a specific attack scenario, an individual aims to shield their identity from being identified by an online FV system used on social media, thereby seeking to evade the system's facial identity-check mechanism. To do so, the attacker needs to create an image that does not match their face. This scenario can be accomplished by creating an empty $\mathcal{M}$atchSet and a $\mathcal{D}$odgeSet that contains the attacker's own face image. This scenario is named None-Targeted Attack, Face De-Identification, or Dodging Attack by the research community~\cite{9464957}.

\textbf{Multi Identity Dodging.} In this scenario, the attacker, who is a wanted criminal, aims to conceal their identity and ensure that the altered image does not resemble any other potentially sensitive identities, such as other criminals. To minimize the likelihood of being recognized as either themselves or another criminal, the attacker must generate an image that deviates from multiple identities, including their own. This can be achieved by defining a $\mathcal{M}$atchSet with no members and a $\mathcal{D}$odgeSet consisting of the images of those identities.

\textbf{Single Identity Impersonation.} 
In another attack scenario, an attacker may wish to gain access to someone else's smartphone by impersonating the victim's identity~\footnote{In this given scenario, we consider having direct access to the smartphone's FV system. Moving forward, a logical progression for this research would involve exploring Physical Attacks, which represent a more realistic setting for this particular scenario.}. To achieve this, the attacker would need to create a set of images that match the victim's face, which can be formulated as a $\mathcal{M}$atchSet with one member being the face image of the victim with an empty $\mathcal{D}$odgeSet. The research community has given the name Targeted Attack to this scenario, as documented in~\cite{9464957}.

\textbf{Multi Identity Impersonation.} 
This situation can arise when the attacker aims to deceive the online FV system of a portal in order to obtain unauthorized access. In this scenario, the attacker possesses incomplete knowledge regarding the authorized employees, lacking awareness of which employees have access privileges and which ones do not. To increase the chances of success, the attacker needs to impersonate all possible individuals who might have proper access. This scenario can be accomplished by creating a $\mathcal{M}$atchSet that contains several members, which are the face images of the employees. In a similar scenario, the attacker may want to gain access to any arbitrary smartphone by fooling its FV system. To achieve this, the attacker needs to create a set of images that look like a large number of identities. This scenario can be formulated as a $\mathcal{M}$atchSet containing face images of \emph{a large number of identities}, while keeping $\mathcal{D}$odgeSet empty. This scenario is an extreme case of the previous one and is known as a Master Face Attack~\cite{nguyen2020generating}.



\textbf{Single Impersonation and Single Dodging.} 
Another scenario arises when we need to satisfy the requirements of both \textbf{Single Identity Dodging} and \textbf{Single Identity Impersonation} \textbf{at the same time}. In a hypothetical scenario, we can envision a situation where a wanted criminal, acting as an attacker, intends to utilize their own image on an online system equipped with an FV mechanism to gain entry. In this case, the attacker seeks to access the system by impersonating an authorized individual. However, the attacker also wants to make sure that their own identity is not recognized. To achieve this, the attacker needs to create a set of images that match the victim's face and do not match their own face. This can be formulated as a $\mathcal{M}$atchSet with one member being the face image of the victim and an $\mathcal{D}$odgeSet with the attacker's own face image. In the same vein, when the attacker intends to hide their identity and prevent the modified image from resembling any other critical identities, the scenario of \textbf{Single Impersonation and Multi Dodging} arises. Additionally, in situations where the attacker has limited information about the individuals they want to impersonate and is unsure of who has access or not, and also needs to avoid being recognized as themselves, the scenario of \textbf{Multi-Impersonation and Single Dodging} arises.


\textbf{Multi Impersonation and Multi Dodging.} Lastly, if an organization allows access to certain individuals, but alerts the police if certain wanted individuals are recognized, the attacker might want to impersonate the authorized individuals and avoid being recognized as one of the wanted individuals. To increase their chance of gaining access, they can use a $\mathcal{M}$atchSet that contains face images of several authorized individuals who might have access, while using a $\mathcal{D}$odgeSet that contains face images of the wanted individuals. This way, the attacker can minimize the risk of getting caught while maximizing the chances of entering the facility.   


## Single Impersonation and Single Dodging Scenario

For this study, we randomly pick a single picture from $\mathcal{F}$ as the $\mathcal{M}$atchSet (with an empty $\mathcal{D}$odgeSet) and aim to modify Albert Einstein's face image to resemble the identity in the selected picture. The experiment was conducted ten times, and the outcome revealed 100\% coverage, implying that in all ten trials, the modified image of Einstein impersonated the randomly chosen person successfully.

In order to address a single dodging scenario, we randomly select an image from $\mathcal{F}$ as the $\mathcal{D}$odgeSet (with an empty $\mathcal{M}$atchSet) and alter it in order to dodge or avoid its identity. Therefore, the image used for the attack is the only member of $\mathcal{D}$odgeSet. The experiment is repeated ten times, and it is found that in ten out of the ten attempts, the dodging was successful.


## Discussion about the $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet on the DodgePersonation Attack

The definition we propose requires a $\mathcal{M}$atchSet and a $\mathcal{D}$odgeSet. One may argue that the use of $\mathcal{D}$odgeSet might be redundant by observing that when matching the identity or identities in $\mathcal{M}$atchSet any other images will be dodged automatically. However, we claim that this might not be true, and thus the use of both $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet is necessary. Let us illustrate this with an example. 
Suppose we face the challenge of creating an image that appears as person A to humans, is recognized by the FV system as person B, and avoids identification as person C (or dodges person C). While some argue that if the FV system confirms the image as person B, it automatically avoids other identities like person C, this assumption is not foolproof. FV systems are imperfect, as demonstrated by the presence of ``master faces'', which means that a solution may match more identities than those in the $\mathcal{M}$atchSet. To address this issue, a $\mathcal{D}$odgeSet is necessary to ensure the robustness and reliability of the system, explicitly considering unwanted identities.



## Exploring the Distribution of Identities in the Embedding Space
Two experiments were performed to gain a better understanding of how the face images of different identities are distributed in the embedding space. The first study involves selecting a random image as the $\mathcal{M}$atchSet (with an empty $\mathcal{D}$odgeSet) and then modifying Albert Einstein's photo to impersonate it. This is repeated five times, then the average result is reported. The process is then repeated for two images, then three, and so on, up to ten images, each time repeating the experiment five times. The outcomes are displayed in Figure~\ref{fig:impersonation_variation}. It is evident that as the number of identities to impersonate increases, the coverage percentage decreases. This suggests that the embeddings are not localized within a specific region of the embedding space, making it impossible for a single point to accurately cover all of them.

\begin{figure}[htpb]
    \centering
    \includegraphics[width=0.45\textwidth]{plots/impersonation_variation.png}
    \caption{Coverage percentage by the number of identities to impersonate using a single image.}
    \label{fig:impersonation_variation}
\end{figure}





## Detailed results of varying size of $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet \label{app:table}

Table~\ref{tab:phase3_10clusters} shows the detailed results of the experiment carried out in Section~\ref{sec:res1} and summarized in Figure~\ref{fig:phase3_10clusters}.



\begin{table}
\centering
\caption{Results of different $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet sizes in Phase 1 and Phase 2 of the One Face to Rule Them All Algorithm. The results are the average of 5 runs for 10 clusters. This table corresponds to the results shown in Figure~\ref{fig:phase3_10clusters}.}
\label{tab:phase3_10clusters}

\resizebox{0.7\textwidth}{!}
{

\begin{tabular}{@{}rrrrrr@{}}
\toprule
 \multirow{2}{*}{$|\mathcal{M}$atchSet$|$} &   \multirow{2}{*}{$|\mathcal{D}$odgeSet$|$} &  \multicolumn{4}{c}{Coverage}\\
  \cline{3-6}
\vspace*{-0.25cm}\\  
  & &  $\overline{\mathcal{M}\text{atchSet}}$ &  $\overline{\mathcal{D}\text{odgeSet}}$ &  $\mathcal{M}$atchSet &  $\mathcal{D}$odgeSet \\
\midrule
         10 &           0 &    100.00 &      0.00 &            100.00 &              0.00 \\ \hline
         10 &           1 &    100.00 &      0.00 &            100.00 &              0.00 \\ \hline
         10 &           2 &    100.00 &      0.00 &            100.00 &              0.00 \\ \hline
         10 &           3 &    100.00 &      0.00 &            100.00 &              0.00 \\ \hline
         10 &          10 &    100.00 &      0.00 &            100.00 &             10.00 \\ \hline
         10 &         100 &    100.00 &      0.00 &            100.00 &              3.25 \\ \hline
         10 &         500 &    100.00 &      0.00 &            100.00 &              4.15 \\ \hline
         10 &        1000 &    100.00 &      0.00 &            100.00 &              3.48 \\ \hline
         10 &        2500 &    100.00 &      0.45 &            100.00 &              2.48 \\ \hline
        100 &           0 &     77.75 &      0.00 &             74.00 &              0.00 \\ \hline
        100 &           1 &     77.00 &      0.00 &             73.50 &              0.00 \\ \hline
        100 &           2 &     77.50 &      0.00 &             74.50 &              0.00 \\ \hline
        100 &           3 &     77.75 &      0.00 &             74.50 &              8.33 \\ \hline
        100 &          10 &     77.25 &      0.00 &             75.25 &             22.50 \\ \hline
        100 &         100 &     76.25 &      0.00 &             74.25 &             13.25 \\ \hline
        100 &         500 &     73.75 &      0.05 &             70.25 &              9.35 \\ \hline
        100 &        1000 &     73.00 &      0.05 &             69.50 &              8.42 \\ \hline
        100 &        2500 &     66.33 &      0.13 &             65.33 &              5.16 \\ \hline
        500 &           0 &     68.85 &      0.00 &             65.15 &              0.00 \\ \hline
        500 &           1 &     66.50 &      0.00 &             62.55 &              0.00 \\ \hline
        500 &           2 &     67.40 &      0.00 &             63.35 &             12.50 \\ \hline
        500 &           3 &     67.70 &      0.00 &             64.10 &             16.67 \\ \hline
        500 &          10 &     67.55 &      5.00 &             63.50 &             20.00 \\ \hline
        500 &         100 &     65.65 &      3.50 &             62.47 &             24.00 \\ \hline
        500 &         500 &     54.40 &      1.15 &             52.35 &             18.55 \\ \hline
        500 &        1000 &     47.60 &      0.88 &             47.40 &             20.38 \\ \hline
        500 &        2500 &     39.20 &      0.45 &             41.40 &             16.67 \\ \hline
       1000 &           0 &     67.62 &      0.00 &             63.82 &              0.00 \\ \hline
       1000 &           1 &     65.42 &      0.00 &             61.52 &              0.00 \\ \hline
       1000 &           2 &     66.12 &      0.00 &             61.78 &             12.50 \\ \hline
       1000 &           3 &     66.45 &      0.00 &             62.68 &              8.33 \\ \hline
       1000 &          10 &     66.85 &      2.50 &             62.38 &             17.50 \\ \hline
       1000 &         100 &     64.47 &     11.33 &             60.70 &             33.67 \\ \hline
       1000 &         500 &     56.00 &      7.65 &             54.68 &             30.20 \\ \hline
       1000 &        1000 &     47.00 &      5.00 &             47.30 &             29.50 \\ \hline
       1000 &        2500 &     35.63 &      3.15 &             40.53 &             26.15 \\ \hline
       2500 &           0 &     65.86 &      0.00 &             61.56 &              0.00 \\ \hline
       2500 &           1 &     64.23 &      0.00 &             59.57 &              0.00 \\ \hline
       2500 &           2 &     64.88 &      0.00 &             61.31 &             12.50 \\ \hline
       2500 &           3 &     65.11 &      0.00 &             60.92 &             25.00 \\ \hline
       2500 &          10 &     65.16 &     12.50 &             61.01 &             50.00 \\ \hline
       2500 &         100 &     64.61 &     30.67 &             60.36 &             46.67 \\ \hline
       2500 &         500 &     61.47 &     29.75 &             58.34 &             41.85 \\ \hline
       2500 &        1000 &     59.25 &     30.20 &             57.29 &             43.60 \\ \hline
       2500 &        2500 &     51.42 &     27.24 &             51.20 &             43.40 \\ 
\bottomrule
\end{tabular}

}
\end{table}

## Extra Results on Multi Identity Impersonation or Master Face \label{app:einstein}

Figure~\ref{fig:einstein2} shows the results of an experimental setting similar to the one described in Section~\ref{sec:einstein} but using a different Source Face image. In this case, we considered a second image of Albert Einstein, which achieved 57.27\% of coverage. 

Experiments similar to the setup in Section~\ref{sec:einstein}, but with 20 different Source Faces (comprising 10 male and 10 female identities), can be seen in Table~\ref{tab:sourcefaces}. The outcomes of these tests are depicted in Figure~\ref{fig:twenty1} and Figure~\ref{fig:twenty2}.


\begin{figure*}[b]
    \centering
    \includegraphics[width=0.8\textwidth]{plots/einstein2.png}
    \caption{One Face to Rule Them All Algorithm for carrying out the Multi Identity Impersonation of 5749 identities using the image of Albert Einstein. The Original image (Source Face) is in a blue box. The remaining images are the Attack Faces, achieving a coverage of 57.27\% of the identities. The previous method covered only 43.82\% of the identities.
    }
    \label{fig:einstein2}
\end{figure*}

\begin{figure*}[b]
    \centering
    \includegraphics[width=0.99\textwidth]{plots/twenty-first.png}
    \caption{One Face to Rule Them All Algorithm for carrying out the Multi Identity Impersonation of 5749 identities using faces 1 to 10 from a pool of 20. Each row's data corresponds to the information presented in Table~\ref{tab:sourcefaces}. The first column displays the Original Image (Source Face), while subsequent columns showcase the Attack Faces.
    }
    \label{fig:twenty1}
\end{figure*}

\begin{figure*}[b]
    \centering
    \includegraphics[width=0.99\textwidth]{plots/twenty-second.png}
    \caption{One Face to Rule Them All Algorithm for carrying out the Multi Identity Impersonation of 5749 identities using faces 11 to 20 from a pool of 20. Each row's data corresponds to the information presented in Table~\ref{tab:sourcefaces}. The first column displays the Original Image (Source Face), while subsequent columns showcase the Attack Faces.
    }
    \label{fig:twenty2}
\end{figure*}




## Fitness Function ablation studies

### Sensitivity of $\mathcal{D}$odgeSet threshold\label{thresh}



Our proposed GA's DodgePersonation Fitness function (cf. Definition~\ref{def:ga}) takes into account two decision thresholds: $th1$ for $\mathcal{M}$atchSet and $th2$ for $\mathcal{D}$odgeSet. In our experiments,  $th1$ and $th2$ are both set to 1.055, as explained in Section~\ref{Evaluation}. In this experiment, we investigate the impact of varying $th2$ on the coverage of $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet.  


Our goal is two-fold: (i) understand if we can generate Attack Faces that are better at dodging the cases in $\mathcal{D}$odgeSet; and (ii) understand the impact of changing threshold $th2$ on the $\mathcal{M}$atchSet coverage. We hypothesize that if the GA is forced to consider a wider margin on the $\overline{\mathcal{D}\text{odgeSet}}$ by increasing $th2$, then more points can be dodged as the optimal points will be further away from this set members. 



We tested the initial value of $th2$ of 1.055 and included four other $th2$ values representing an increase of $th2$ by 3\%, 4\%, 5\%, and 6\%. We repeated these experiments 5 times using 1000 and 500 identities randomly selected for the $\mathcal{M}$atchSet and the $\mathcal{D}$odgeSet, respectively. The average coverage results on $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet for Phase 1 and Phase 2 are displayed in Table~\ref{tab:nthresh}. 



In Phase 1 and Phase 2, the coverage of the $\mathcal{D}$odgeSet when using the default value of 1.055 is 7.65\% and 30.20\%, respectively. Our special purpose GA was not able to avoid 7.65\% of the $\overline{\mathcal{D}\text{odgeSet}}$ members in the embedding space while the generated Attack Faces could not dodge 30.2\% of the $\mathcal{D}$odgeSet cases. 
Increasing the $\mathcal{D}$odgeSet $th2$ by 3\% led to a significant decrease in coverage percentages of the $\mathcal{D}$odgeSet in both phases, which confirms that increasing the threshold $th2$ helps to keep the Attack Faces further away from the $\mathcal{D}$odgeSet cases. However, we observe that the $\mathcal{M}$atchSet coverage is negatively impacted by this change, showing a decreased coverage. The $\mathcal{M}$atchSet coverage continues to decrease as the threshold $th2$ increases while the coverage of the $\mathcal{D}$odgeSet tends to zero. After an increase of 4\% in the $th2$ only the $\mathcal{M}$atchSet coverage is being affected because the $\mathcal{D}$odgeSet coverage is already very close to the ideal value of zero. Therefore, we confirm that adjusting the $\mathcal{D}$odgeSet threshold ratio can help to achieve better dodging results while experiencing lower impersonation results. This trade-off should be taken into account based on the problem's nature and the importance of dodging versus impersonation.

\begin{table}
\centering
\caption{Coverage results for different $\mathcal{D}$odgeSet thresholds $th2$ on Phases 1 and 2 (Phase 2 results in parenthesis). }\label{tab:nthresh}
\resizebox{0.4\textwidth}{!}{
\begin{tabular}{@{}cccc@{}}

\toprule
$th2$& increase \% &
$\mathcal{M}$atchSet Cov. & $\mathcal{D}$odgeSet Cov.\\
\midrule
1.055&  -                     & 56.00 (54.68)                                         & 7.65 (30.20)                                      \\
1.086 & 3\%                & 41.56 (42.30)                                      & 0.00 (3.04)                                       \\
1.097 & 4\%                & 37.98 (39.50)                                      & 0.00 (1.48)                                       \\
1.107 & 5\%                 & 30.46 (35.62)                                        & 0.00 (0.60)                                       \\
1.118 & 6\%                 & 31.02 (33.72)                                      & 0.00 (0.40)                                       \\
\bottomrule
% \end{tblr}
\end{tabular}
}

\end{table}



## Phase 2 - An Alternative the genetic algorithm: Projected Mean 
It is possible that the average of all the $\overline{\mathcal{M}\text{atchSet}_i}$ members is close enough to most of its members (however, there is no guarantee that it is far enough from the $\overline{\mathcal{D}\text{odgeSet}}$ members). Constrained by embedding space, if the search space is the surface of a hyper-sphere, the average point of some disjoint points will land inside it. In such embedding spaces, we project the average point back onto the hyper-sphere's surface. Table~\ref{PM_GA} indicates projected mean yields good candidate points when $\overline{\mathcal{D}\text{odgeSet}}$ is empty. The contrast is more clear in difficult cases where $\overline{\mathcal{D}\text{odgeSet}}$ is not empty. Therefore, the genetic algorithm is works much better than simple projection of the average of $\overline{\mathcal{M}\text{atchSet}}$ onto the unit hyper-sphere. 

\begin{table}
\centering
\caption{The results of projected mean and genetic algorithm search for $\mathcal{M}$atchSet of size {100} and $\mathcal{D}$odgeSet size of size {0, 100, 500}. The results are the average of 5 runs for 10 clusters.}
\label{PM_GA}
\resizebox{\columnwidth}{!}
{
\begin{tabular}{rrlrr}
\toprule
 $\lvert \mathcal{M}\text{atchSet} \rvert $ &  $\lvert  \mathcal{D}\text{odgeSet} \rvert $ &                 method (Phase 2)&  $\mathcal{M}$atchSet coverage(\%) &  $\mathcal{D}$odgeSet coverage(\%) \\ 
\midrule
        100 &           0 &             GA &              77.75 &               0.00 \\ 
        100 &           0 & Projected Mean &              76.50 &               0.00 \\ 
        100 &         100 &             GA &              76.25 &               0.00 \\ 
        100 &         100 & Projected Mean &              76.50 &              29.25 \\ 
        100 &         500 &             GA &              73.75 &               0.05 \\ 
        100 &         500 & Projected Mean &              76.50 &              29.60 \\ \bottomrule
\end{tabular}
}
\end{table}



## Sensitivity of parameter $\gamma$
Parameter $\gamma$ in the GA Fitness function weights the $DPloss$ of the $\overline{\mathcal{M}\text{atchSet}}$ and $\overline{\mathcal{D}\text{odgeSet}}$. We randomly selected 1000 $\overline{\mathcal{M}\text{atchSet}}$ members and 500 $\overline{\mathcal{D}\text{odgeSet}}$ members and tested the values of 0, 0.1, 0.3, 0.5, 0.7, 0.9, and 1 for parameter $\gamma$. We repeated these experiments five times and reported the average. Figure~\ref{fig:gamma} shows the results of these experiments. 

We observe that, when $\gamma$ is 0, the focus is entirely on evading $\overline{\mathcal{D}\text{odgeSet}}$ members, leading to the neglect of $\overline{\mathcal{M}\text{atchSet}}$ members. On the other hand, when $\gamma$ is 1, the coverage of $\overline{\mathcal{M}\text{atchSet}}$ members peaks, while $\overline{\mathcal{D}\text{odgeSet}}$ members are ignored during dodging, resulting in a coverage of a significant percentage of $\overline{\mathcal{D}\text{odgeSet}}$ members. With an increase in $\gamma$ to 0.1, $\overline{\mathcal{M}\text{atchSet}}$ members gain considerable coverage while the coverage of $\overline{\mathcal{D}\text{odgeSet}}$ does not increase significantly. As $\gamma$ continues to grow, the emphasis on $\overline{\mathcal{M}\text{atchSet}}$ increases, resulting in coverage of more members, while the emphasis on $\overline{\mathcal{D}\text{odgeSet}}$ decreases, leading to less dodging of $\overline{\mathcal{D}\text{odgeSet}}$ members. Overall, the results are fairly stable for values of $\gamma$ between 0.1 and 0.9.  


\begin{figure}[!t]
    \centering
    \includegraphics[width=0.3\textwidth]{plots/gamma.png}
    \caption{Impact of $\gamma$ on the coverage of $\overline{\mathcal{M}\text{atchSet}}$ and $\overline{\mathcal{D}\text{odgeSet}}$.}
    \label{fig:gamma}
\end{figure}






## Measuring the attack difficulty \label{sec:difficulty}
Given the FV system robustness and the identities and number of images selected for $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet, the DodgePersonation Attack can be more difficult or easy to solve. Based on the above taxonomy and definitions, we defined three measures to assess the difficulty of the system. 

\begin{definition}[System Difficulty Measure]\label{def:sys-dif}
Given $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet, such that $\lvert \mathcal{M}\text{atchSet}\rvert=k$ and $\lvert \mathcal{D}\text{odgeSet}\rvert=l$, we define the system difficulty as:

SDiff =
\[
% SDiff =
\begin{cases} 
        \frac{\Bigg\lvert\sum \limits_{\substack{\forall m \in \mathcal{M}\text{atchSet}\\  d \in \mathcal{D}\text{odgeSet}}} Dist(FM(m))=Dist(FM(d))\Bigg\rvert}{k \times l} & \text{if } k>0 \wedge l>0\\
        0 & \text{otherwise}\\
        
    \end{cases}
\]
\end{definition}

\begin{definition}[System Positive Difficulty Measure]\label{def:sys-dif-pos}
Given $\mathcal{M}$atchSet such that $\lvert \mathcal{M}\text{atchSet}\rvert=k$, we define the system positive difficulty as:

PDiff =
\[
% PDiff =
\begin{cases} 
        \frac{\Bigg\lvert \sum \limits_{\substack{\forall m_i, m_j \in \mathcal{M}\text{atchSet}}} Dist(FM(m_i))=Dist(FM(m_j))\Bigg\rvert}{\binom{m}{2}} & \text{if } k>0 \\
        0 & \text{otherwise}\\
        
    \end{cases}
\]
\end{definition}




\begin{definition}[System Negative Difficulty Measure]\label{def:sys-dif-neg}
Given $\mathcal{D}$odgeSet such that $\lvert \mathcal{D}\text{odgeSet}\rvert=l$, we define the system positive difficulty as:

NDiff =
\[
% NDiff =
\begin{cases} 
        \frac{\Bigg\lvert \sum \limits_{\substack{\forall d_i, d_j \in \mathcal{D}\text{odgeSet}}} Dist(FM(d_i))=Dist(FM(d_j))\Bigg\rvert}{\binom{l}{2}} & \text{if } l>0 \\
        0 & \text{otherwise}\\
        
    \end{cases}
\]
\end{definition}
These metrics can measure how easy it is to attack a given system while considering the selected $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet. Namely, a biased selection of $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet can impact the attacks success rate by increasing or decreasing it. Thus, these measures can help us to assess the deployment setting. 

The defined system difficulty measures vary between zero and one. 
When $SDiff$ is closer to 1, this means that all images in the $\mathcal{M}$atchSet are close to all images in the $\mathcal{D}$odgeSet, and thus, finding the existence of a solution will tend to zero. This happens because maximizing number of the faces matched in the $\mathcal{M}$atchSet will inhibit dodging the faces in the $\mathcal{D}$odgeSet because they are close to each other. When $SDiff$ tends to zero, there is no matching between the $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet. In this case, there is no guarantee about how easy or difficult it is to attack the system because it will depend on the distribution and characteristics of the images in $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet.


When $PDiff$ tends to one, then all images in $\mathcal{M}$atchSet are close to each other, which means that the possibility of finding an image close to all of them also increases. On the other hand, when $PDiff$ tends to zero, then finding a single image that matches all images in $\mathcal{M}$atchSet will be harder. In this case, multiple images may be required in order to collectively match the identities in $\mathcal{M}$atchSet.

Finally, when $NDiff$ tends to one, this means that the images in $\mathcal{D}$odgeSet are close to each other which makes the goal of dodging them easier. The reverse happens when $NDiff$ tends to zero.



## System Difficulty Measurement
Tables~\ref{tab:system-difficulty-random} and \ref{tab:system-difficulty-male-female} present the results of tests performed based on the definitions of system difficulty measure or $SDiff$, (cf. Equation~\ref{def:sys-dif}), positive system difficulty measure or $PDiff$ (cf. Equation~\ref{def:sys-dif-pos}), and negative system difficulty measure or $NDiff$ (cf. Equation~\ref{def:sys-dif-neg}). The first table displays results for randomly selected members in the $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet, while the second table displays results for male members in the $\mathcal{M}$atchSet and female members in the $\mathcal{D}$odgeSet.


\begin{table}[ht]
\centering
\caption{Average system difficulty measurements for $\mathcal{M}$atchSet and $\mathcal{D}$odgeSet with random identities, based on 5 runs.}
\begin{tblr}{|c|c|X[c]|X[c]|X[c]|}
\hline
$\mathcal{M}$atchSet size & $\mathcal{D}$odgeSet size & System Difficulty Measure & System Positive Difficulty Measure & System Negative Difficulty Measure \\
\hline
100 & 100 & 0.74\% & 0.75\% & 0.69\% \\
200 & 100 & 0.82\% & 0.73\% & 0.91\% \\
300 & 100 & 0.82\% & 0.79\% & 0.84\% \\
\hline
\end{tblr}
\label{tab:system-difficulty-random}
\end{table}

\begin{table}[ht]
\caption{System difficulty measurements for $\mathcal{M}$atchSet with male identities and $\mathcal{D}$odgeSet with female identities.}
\centering
\begin{tblr}{|c|c|X[c]|X[c]|X[c]|}
\hline
$\mathcal{M}$atchSet size & $\mathcal{D}$odgeSet size & System Difficulty Measure & System Positive Difficulty Measure & System Negative Difficulty Measure \\
\hline
100 & 100 & 0.05\% & 1.30\% & 1.15\% \\
200 & 100 & 0.05\% & 1.27\% & 1.15\% \\
300 & 100 & 0.04\% & 1.18\% & 1.15\% \\
\hline
\end{tblr}
\label{tab:system-difficulty-male-female}
\end{table}


## Pseudo-Code for the DodgePersonation Attack (Phase 1)
Algorithm~\ref{alg:phase2} describes the procedural steps for executing the DodgePersonation Attack by navigating through the embedding space to identify a point or a set of points.

\begin{algorithm}[htpb]
\footnotesize
% \small
\caption{One Face to Rule Them All - Phase 1: Embedding space Search}\label{alg:phase2}
  \KwInput{$\overline{\mathcal{M}\text{atchSet}}$: set of all face embeddings to impersonate;\\
  \hspace{0.8cm}$\overline{\mathcal{D}\text{odgeSet}}$: set of all face embeddings to dodge;\\
  \hspace{0.8cm}$\alpha$, $\beta$: weights for positive and negative $DPloss$, respectively;\\
  \hspace{0.8cm}$\gamma$: weight for the DodgePersonation Fitness function\\
  \hspace{0.8cm}$th1$, $th2$: DodgePersonation Fitness thresholds\\
  \hspace{0.8cm}$gen$: number of GA generations\\
  \hspace{0.8cm}$C$: number of clusters 
  }
  \KwOutput{$best\_emb$: best C points found in the embedding space. }
  
 
 $best\_emb \gets []$\\
 $Clusters \gets K\text{-}Means(\textrm{ $C$, $\overline{\mathcal{M}\text{atchSet}}$)}$\\
\For{cluster in Clusters}{ 

    $centroid_{norm} \gets\textrm{cluster centroid normalized to unit vector}$\\
    $pp \gets\textrm{ initialize first population with }centroid_{norm}$\\
    \For{generation in gen}{
    $pp'\gets \text{ generate new population based on } pp$\\
    $pp'_{norm} \gets\textrm{normalize to unit vector elements of } pp'$\\
    $loss \gets\textrm{[]}$\\
    \For{$pp'_{i}$ in $pp'_{norm}$ }{
        $loss \gets loss \cup  fitness(pp'_{i},cluster,\overline{\mathcal{D}\text{odgeSet}},th1,th2,\alpha,\beta,\gamma)$
    }

    $pp \gets $ top $|pp'|$ with lowest $loss$ from $pp'_{norm}$ 
} 
    $best\_emb \gets best\_emb \cup \textrm{embedding with lowest loss in } pp$
}
\Return $best\_emb$
\end{algorithm}


