import nox

@nox.session
def tests(session):
    session.install('pytest')
    session.run('-m pytest tests/')

@nox.session
def lint(session):
    session.install('flake8')
    session.run('flake8')

@nox.session
def black(session):
    session.install('black')
    session.run('black')

@nox.session
def mypy(session):
    session.install('mypy')
    session.run('mypy --ignore-missing-imports')