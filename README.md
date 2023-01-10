# marcelcastrobr.github.io


My github homepage using [HUGO](https://github.com/gohugoio/hugo).

Github actions are done using [Hugo-pages-template](https://github.com/pages-hugoconf-2022/hugo-pages-template).



Simple steps to test it.

- Clone the repository

- Install HUGO. In my case, I use macOS and install command was:

  ```bash
  $ brew install hugo
  ```

- Run Hugo server locally to debug the website.

  ```bash
  $ hugo server
  ```



Note: I am using "canonifyURLs = true"  in config.toml in order to make sure images are visible when using GitHub-pages (ref. Topic discussion [here](https://discourse.gohugo.io/t/images-not-showing-on-github-with-ananke-theme/26776)).
