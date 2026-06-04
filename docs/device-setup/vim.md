---
sidebar_position: 2
title: Configure Vim
description: Personal Vim configuration for Yi-Chen Zhang
---

# Vim Configuration

Before applying this configuration, make sure you have a full-featured Vim installed. This is my personal Vim configuration located at `/etc/vim/vimrc`.

```vim
" All system-wide defaults are set in $VIMRUNTIME/debian.vim and sourced by
" the call to :runtime you can find below.  If you wish to change any of those
" settings, you should do it in this file (/etc/vim/vimrc), since debian.vim
" will be overwritten everytime an upgrade of the vim packages is performed.
" It is recommended to make changes after sourcing debian.vim since it alters
" the value of the 'compatible' option.
runtime! debian.vim

" Vim will load $VIMRUNTIME/defaults.vim if the user does not have a vimrc.
" This happens after /etc/vim/vimrc(.local) are loaded, so it will override
" any settings in these files.
" If you don't want that to happen, uncomment the below line to prevent
" defaults.vim from being loaded.
let g:skip_defaults_vim = 1

" Uncomment the next line to make Vim more Vi-compatible
" NOTE: debian.vim sets 'nocompatible'.  Setting 'compatible' changes numerous
" options, so any other options should be set AFTER setting 'compatible'.
"set compatible

" Vim5 and later versions support syntax highlighting. Uncommenting the next
" line enables syntax highlighting by default.
if has("syntax")
  syntax on
endif

" If using a dark background within the editing area and syntax highlighting
" turn on this option as well
set background=dark

" Uncomment the following to have Vim jump to the last position when
" reopening a file
au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif

" Uncomment the following to have Vim load indentation rules and plugins
" according to the detected filetype.
filetype plugin indent on

" OPTIONAL: Starting with Vim 7, the filetype of empty .tex files defaults to
" 'plaintex' instead of 'tex', which results in vim-latex not being loaded.
" The following changes the default filetype back to 'tex':
let g:tex_flavor='latex'

" TIP: if you write your \label's as \label{fig:something}, then if you
" type in \ref{fig: and press <C-n> you will automatically cycle through
" all the figure labels. Very useful!
set iskeyword+=:

" The following are commented out as they cause vim to behave a lot
" differently from regular Vi. They are highly recommended though.
"set showcmd    " Show (partial) command in status line.
"set showmatch  " Show matching brackets.
"set ignorecase " Do case insensitive matching
"set smartcase  " Do smart case matching
"set incsearch  " Incremental search
"set autowrite  " Automatically save before commands like :next and :make
"set hidden     " Hide buffers when they are abandoned
"set mouse=a    " Enable mouse usage (all modes)

set clipboard^=unnamed

" configure expanding of tabs for various file types
au BufRead,BufNewFile *.py  set expandtab
au BufRead,BufNewFile *.c   set expandtab
au BufRead,BufNewFile *.cpp set expandtab
au BufRead,BufNewFile *.h   set expandtab
au BufRead,BufNewFile Makefile* set noexpandtab

set expandtab          " enter spaces when tab is pressed
"set textwidth=80      " break lines when line length increases
set tabstop=2          " use 2 spaces to represent tab
set softtabstop=2      " number of spaces a tab counts for (under editing operation)
set shiftwidth=2       " number of spaces to use for auto indent
set autoindent         " copy indent from current line when starting a new line
set backspace=indent,eol,start  " make backspaces more powerful
set ruler              " show line and column number
set showcmd            " show (partial) command in status line

autocmd bufreadpre *.txt setlocal textwidth=80

colorscheme default

hi DiffAdd    gui=none  guifg=NONE    guibg=#bada9f
hi DiffChange gui=none  guifg=NONE    guibg=#e5d5ac
hi DiffDelete gui=bold  guifg=#ff8080 guibg=#ffb0b0
hi DiffText   gui=none  guifg=NONE    guibg=#8cbee2

highlight ExtraWhitespace ctermbg=green guibg=green
au FileType cpp match ExtraWhitespace /\s\+$/

" Source a global configuration file if available
if filereadable("/etc/vim/vimrc.local")
  source /etc/vim/vimrc.local
endif
```

:::tip
Vim Copy Across Terminals
The `clipboard^=unnamed` setting requires a Vim build with clipboard support. If copy/paste across terminals isn't working, install a clipboard-capable build:
:::

```bash
sudo apt install vim-gtk3
```

:::note
The package name may differ across distributions (e.g. `vim-gnome`, `vim-gtk3`, `vim-X11`). Check your distro's package manager for the available options.
:::

You can verify clipboard support is enabled with:

```bash
vim --version | grep clipboard
```

You should see `+clipboard` in the output.
