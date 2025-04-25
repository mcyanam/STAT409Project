% Define a custom class for a concise article-like format
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{psu-plan}[2025/04/01 v1.0 Custom class for PSU Plan]

% Load base class
\LoadClass[a4paper,10pt]{article}

% Set default font (professional and accessible)
\RequirePackage{fontspec}
\setmainfont{Times New Roman}

% Geometry and layout
\RequirePackage[margin=1in]{geometry}

% Package imports for modern styling
\RequirePackage{titling}
\RequirePackage{hyperref}
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{enumitem}

% Hyperref setup
\hypersetup{
    breaklinks=true,
    colorlinks=true,
    linkcolor=black,
    citecolor=black,
    urlcolor=blue,
    pdfproducer={LaTeX},
    pdfcreator={pdfLaTeX},
    pdfpagemode=UseNone,
    pdfstartview=FitH,
    pdfnewwindow=true
}

%% Title setup
\renewcommand{\maketitle}{
    \begin{center}
        {\Large\bfseries \@title \par}
        \vspace{0.5cm}
        {\large \@author \par}
        \vspace{0.5cm}
        {\small \@date \par}
    \end{center}
    \vspace{0.5cm}
    \hrule
    \vspace{0.5cm}
}

%% Section formatting
\renewcommand{\section}{\@startsection{section}{1}{0pt}%
    {1.5ex plus 1ex minus .2ex}%
    {1.5ex plus .2ex}%
    {\normalfont\large\bfseries}}

\renewcommand{\subsection}{\@startsection{subsection}{2}{0pt}%
    {1.25ex plus 1ex minus .2ex}%
    {1.25ex plus .2ex}%
    {\normalfont\normalsize\bfseries}}

\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{0pt}%
    {1ex plus 1ex minus .2ex}%
    {1ex plus .2ex}%
    {\normalfont\normalsize\itshape}}

%% Custom commands for metadata
\newcommand{\professor}[1]{\def\@professor{#1}}
\newcommand{\client}[1]{\def\@client{#1}}

\providecommand{\@professor}{}
\providecommand{\@client}{}

%% Custom footer
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small \@title}
\fancyhead[R]{\small \@author}
\fancyfoot[C]{\thepage}

%% Caption setup
\captionsetup{
    font={small},
    labelfont={bf},
    labelsep=period,
    justification=centering
}

%% Abstract environment
\newenvironment{customabstract}{
    \section*{Abstract}
    \itshape
}{\par}

%% References section
\newcommand{\references}[1][References]{
    \section*{#1}
    \addcontentsline{toc}{section}{#1}
}

%% Document start and metadata setup
\AtBeginDocument{
    \hypersetup{
        pdftitle={\@title},
        pdfauthor={\@author},
        pdfkeywords={\@professor, \@client}
    }
}

\endinput
