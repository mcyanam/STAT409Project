\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{psu-report}[2025/04/11 Portland State University Report Class]

%% Base class
\LoadClass[12pt,letterpaper,oneside]{memoir}

%% Required packages
\RequirePackage{fontspec}
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{xcolor}
\RequirePackage{caption}
\RequirePackage{etoolbox}
\RequirePackage{lipsum} % For sample text

%% Set up geometry (1-inch margins as required)
\geometry{
  letterpaper,
  margin=1in,
  includeheadfoot
}

\newcommand{\doctype}[1]{\def\@doctype{#1}}
\doctype{report}  % Default value

\newcommand{\psumonth}[1]{\def\@month{#1}}
\newcommand{\psuyear}[1]{\def\@year{#1}}

\newcommand{\professorOne}[1]{\def\@professorOne{#1}}
\newcommand{\professorTwo}[1]{\def\@professorTwo{#1}}
\newcommand{\professorThree}[1]{\def\@professorThree{#1}}
\newcommand{\professorFour}[1]{\def\@professorFour{#1}}
\newcommand{\clientOne}[1]{\def\@clientOne{#1}}
\newcommand{\clientTwo}[1]{\def\@clientTwo{#1}}
\newcommand{\clientThree}[1]{\def\@clientThree{#1}}
\newcommand{\clientFour}[1]{\def\@clientFour{#1}}
\providecommand{\@professorOne}{}
\providecommand{\@professorTwo}{}
\providecommand{\@professorThree}{}
\providecommand{\@professorFour}{}
\providecommand{\@clientOne}{}
\providecommand{\@clientTwo}{}
\providecommand{\@clientThree}{}
\providecommand{\@clientFour}{}


\newcommand{\documentformat}[1]{\def\@documentformat{#1}}
\documentformat{monograph} % Default is monograph

%% Load hyperref last to avoid conflicts
\RequirePackage{hyperref}
\hypersetup{
    breaklinks=true,
    colorlinks=true,
    linkcolor=black,
    citecolor=black,
    filecolor=black,
    urlcolor=blue,
    pdfproducer={LaTeX},
    pdfcreator={pdfLaTeX},
    pdfpagemode=UseNone,
    pdfstartview=FitH,
    pdfnewwindow=true
}

%% Update PDF metadata at begin document
\AtBeginDocument{
  \hypersetup{
    pdftitle={\@title},
    pdfauthor={\@author},
    pdfkeywords={Portland State University, \@doctype}
  }
}

%% Double spacing by default - use memoir class commands
\OnehalfSpacing

%% Title Page 
\renewcommand{\maketitle}{
    \begin{titlingpage}
        \begin{center}
            \vspace*{1in}
            {\Large\bfseries \@title \par}
            \vspace{2cm}
            by\\
            \vspace{1cm}
            {\large \@author}

            \vfill

            \begin{center}
                \begin{minipage}{0.45\textwidth}
                    \raggedright
                    \MakeUppercase{Profesors}\\
                    \@professorOne\\ 
                    \ifdefempty{\@professorTwo}{}{{\@professorTwo\\ }}
                    \ifdefempty{\@professorThree}{}{{\@professorThree\\ }}
                    \ifdefempty{\@professorFour}{}{{\@professorFour\\ }}
                \end{minipage}
                \hfill
                \begin{minipage}{0.45\textwidth}
                    \raggedleft
                    \MakeUppercase{Client}\\
                    \@clientOne\\ 
                    \ifdefempty{\@clientTwo}{}{{\@clientTwo\\ }}
                    \ifdefempty{\@clientThree}{}{{\@clientThree\\ }}
                    \ifdefempty{\@clientFour}{}{{\@clientFour\\ }}
                \end{minipage}
            \end{center}

            
            \vfill
            
            Portland State University\\
            \@month{} \@year{}\\ 
        \end{center}
    \end{titlingpage}
}

%% Copyright page
\newcommand{\makecopyright}{
    \thispagestyle{empty}
    \null
    \vfill
    \begin{center}
        © \@year{} \@author
    \end{center}
    \vfill
    \clearpage
}

%% Abstract environment (renamed to avoid conflict with memoir)
\newenvironment{psuabstract}{
    \clearpage
    \phantomsection
    \addcontentsline{toc}{chapter}{Abstract}
    \begin{center}
        \Large \textbf{Abstract}
    \end{center}
    \setcounter{page}{1}
    \pagenumbering{roman}
}{\clearpage}

%% Table of contents setup
\renewcommand{\contentsname}{Table of Contents}
\setlength{\cftchapternumwidth}{2em}

%% Dedication environment
\newenvironment{psudedication}{
    \clearpage
    \phantomsection
    \addcontentsline{toc}{chapter}{Dedication}
    \begin{center}
        \Large \textbf{Dedication}
    \end{center}
    \begin{center}
        \begin{itshape}
}{\end{itshape}\end{center}\clearpage}

%% Acknowledgments environment
\newenvironment{psuacknowledgments}{
    \clearpage
    \phantomsection
    \addcontentsline{toc}{chapter}{Acknowledgments}
    \begin{center}
        \Large \textbf{Acknowledgments}
    \end{center}
}{\clearpage}

%% Glossary environment (renamed to avoid conflict with memoir)
\newenvironment{psuglossary}{
    \clearpage
    \phantomsection
    \addcontentsline{toc}{chapter}{Glossary}
    \begin{center}
        \Large \textbf{Glossary}
    \end{center}
}{\clearpage}

%% Preface environment
\newenvironment{psupreface}{
    \clearpage
    \phantomsection
    \addcontentsline{toc}{chapter}{Preface}
    \begin{center}
        \Large \textbf{Preface}
    \end{center}
}{\clearpage}

%% Chapter heading page for multi-paper format
\newcommand{\chapterheadingpage}[2]{
    \clearpage
    \phantomsection
    \chapter{#1}
    \begin{center}
        \textit{#2}
    \end{center}
}

%% Command to start the main body
\newcommand{\startbody}{
    \clearpage
    \pagenumbering{arabic}
    \setcounter{page}{1}
}

%% Figure and table caption setup
\captionsetup{
    font={small},
    labelfont={bf},
    labelsep=period,
    margin=0.5in,
    justification=justified
}

%% Reference section
\newcommand{\references}[1][References]{
    \chapter*{#1}
    \addcontentsline{toc}{chapter}{#1}
    \markboth{#1}{#1}
}

%% Set default font (professional and accessible)
\setmainfont{Times New Roman}

\endinput
