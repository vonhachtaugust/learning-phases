# Machine learning phases of matter

Topological order is studied using neural networks.

## Getting Started

Here's how to set up `learning-phases` for local development.

1. Fork the `learning-phases` repo on GitHub.
2. Clone your fork locally::

    ```bash
    $ git clone git@github.com:your_name_here/learning-phases.git
    ```

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development

    ```bash
    $ mkvirtualenv $HOME/.virtualenvs/learning-phases
    $ source $HOME/.virtualenvs/learning-phases/bin/activate
    $ pip install -r learning-phases/requirements_dev.txt
    ```

4. Create a branch for local development

    ```bash
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the tests,

    ```bash
    $ cd learning-phases/
    $ nosetests
    ```

6. Commit your changes and push your branch to GitHub::

    ```bash
    $ git add .
    $ git commit -m "Your detailed description of your changes"
    $ git push origin master name-of-your-bugfix-or-feature
    ```
    
7. Submit a pull request through the GitHub website.

## Contributing

Feel free to contributing. As for now, there are no specific requirements for contributing.

## Authors

* **August von Hacht** - *Initial work* - [vonhachtaugust](https://github.com/vonhachtaugust)

See also the list of [contributors](https://github.com/vonhachtaugust/learning-phases/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

