import React, { Component } from 'react';

import { withRouter } from 'react-router-dom';

import AuthForm from 'components/AuthForm';
import OAuth2Buttons from 'components/OAuth2Buttons';
import Spinner from 'components/Spinner';

import styles from './index.module.css';

import EmailImage from 'assets/images/Email.svg';
import NameImage from 'assets/images/Name.svg';
import PasswordImage from 'assets/images/Password.svg';

class Auth extends Component {
    state = {
        loading: false,
        error: null,
        showLogin: true,
        authMessage: null,
        messageColor: null
    };

    toggleFormsHandler = () => {
        this.setState(prevState => {
            return {
                showLogin: !prevState.showLogin,
                authMessage: null
            };
        })
    };

    authHandler = async credentials => {
        this.setState({ loading: true });
        const path = this.state.showLogin ? process.env.REACT_APP_LOGIN_PATH : process.env.REACT_APP_SIGNUP_PATH;
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(credentials)
        };
        let response, data;
        try {
            response = await fetch(process.env.REACT_APP_BASE_URL + path, options);
            data = await response.json();
        } catch (err) {
            this.setState({
                loading: false,
                error: err
            });
        }

        let authMessage, messageColor, showLogin = true;
        switch (response.status) {
            case 201:
                if (!this.state.showLogin) {
                    authMessage = 'Account created successfully';
                    messageColor = 'green';
                }
                break;
            case 200:
                if (this.state.showLogin) {
                    this.props.login(data.token);
                    return this.props.history.push('/');
                }
                break;
            case 401:
                if (this.state.showLogin) {
                    authMessage = 'Invalid credentials';
                    messageColor = 'red';
                }
                break;
            case 422:
                if (!this.state.showLogin) {
                    authMessage = 'Invalid data';
                    messageColor = 'red';
                    showLogin = false;
                }
                break;
            default:
                authMessage = 'Error has happened';
                messageColor = 'red';
        }

        this.setState({
            loading: false,
            showLogin: showLogin,
            authMessage: authMessage,
            messageColor: messageColor
        })
    };

    render() {
        if (this.state.error) { throw this.state.error; }

        const position = this.state.showLogin ? styles.Left : styles.Right;

        const loginInputs = {
            email: {
                placeholder: 'Email',
                type: 'email',
                value: '',
                isValid: false,
                isTouched: false,
                errorMessage: null,
                background: EmailImage,
                validation: {}
            },
            password: {
                placeholder: 'Password',
                type: 'password',
                value: '',
                isValid: false,
                isTouched: false,
                errorMessage: null,
                background: PasswordImage,
                validation: {}
            }
        };

        const signupInputs = {
            name: {
                placeholder: 'Name',
                type: 'text',
                value: '',
                isValid: false,
                isTouched: false,
                errorMessage: null,
                background: NameImage,
                validation: {
                    isRequired: true,
                    minLength: 4,
                    maxLength: 64
                }
            },
            email: {
                placeholder: 'Email',
                type: 'email',
                value: '',
                isValid: false,
                isTouched: false,
                errorMessage: null,
                background: EmailImage,
                validation: {
                    isRequired: true,
                    minLength: 4,
                    maxLength: 64,
                    regex: /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/
                }
            },
            password: {
                placeholder: 'Password',
                type: 'password',
                value: '',
                isValid: false,
                isTouched: false,
                errorMessage: null,
                background: PasswordImage,
                validation: {
                    isRequired: true,
                    minLength: 8,
                    maxLength: 64
                }
            }
        };

        let authContent = <Spinner theme="dark" />;
        if (!this.state.loading) {
            authContent = (
                <>
                    <div className={`${styles.FormContainer} ${position}`}>
                        <div className={this.state.showLogin ? styles.Shown : styles.Hide}>
                            <AuthForm
                                inputs={loginInputs}
                                submit={this.authHandler}
                                heading="Log in to your account"
                                message={this.state.authMessage}
                                messageColor={this.state.messageColor}
                                buttonContent="LOG IN"
                                switch={this.toggleFormsHandler}
                                switchContent="Signup"
                            />
                        </div>
                        <div className={this.state.showLogin ? styles.Hide : styles.Shown}>
                            <AuthForm
                                inputs={signupInputs}
                                submit={this.authHandler}
                                heading="Create new account"
                                message={this.state.authMessage}
                                messageColor={this.state.messageColor}
                                buttonContent="SIGN UP"
                                switch={this.toggleFormsHandler}
                                switchContent="Login"
                            />
                        </div>
                    </div>
                    <OAuth2Buttons
                        switch={this.toggleFormsHandler}
                        switchButtonContent="Log in"
                    />
                    <OAuth2Buttons
                        switch={this.toggleFormsHandler}
                        switchButtonContent="Signup"
                    />
                </>
            );
        }

        return (
            <main className={styles.Auth}>
                {authContent}
            </main>
        );
    }
}

export default withRouter(Auth);
