import React, { Component } from 'react';

import AuthForm from 'components/AuthForm';
import Logo from 'components/Logo';
import Spinner from 'components/Spinner';

import styles from './index.module.css';

import EmailImage from 'assets/images/Email.png';
import NameImage from 'assets/images/Name.png';
import PasswordImage from 'assets/images/Password.png';

class Auth extends Component {
    state = {
        loading: false,
        showLogin: 'Login',
        authMessage: null,
        messageColor: null
    };

    toggleFormsHandler = () => {
        this.setState(prevState => {
            return {
                showLogin: !prevState.showLogin
            };
        })
    };

    authHandler = async values => {
        this.setState({ loading: true });
        const path = this.state.showLogin ? process.env.REACT_APP_LOGIN_PATH : process.env.REACT_APP_SIGNUP_PATH;
        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(values)
        };
        let response;
        try {
            response = await fetch(process.env.REACT_APP_BASE_URL + path, options);
        } catch (err) {
            console.log(err);
        }
        console.log(response);
        this.setState({
            loading: false,
            showLogin: true,
            authMessage: 'Account created successfully',
            messageColor: 'green'
        })
    };

    render() {
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
                    maxLength: 60
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
                    maxLength: 60
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
                    minLength: 4,
                    maxLength: 60
                }
            }
        };

        let authContent = <Spinner />;
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
                    <div className={styles.LoginToggle}>
                        <button onClick={this.toggleFormsHandler} className={styles.ToggleButton}>
                            Log in or continue with social account
                        </button>
                    </div>
                    <div className={styles.Oauth2Login}>
                        <h2>Login with social account</h2>
                        <button className={styles.Oauth2LoginButton}>
                            <Logo src="Google" />
                        </button>
                        <button className={styles.Oauth2LoginButton}>
                            <Logo src="Facebook" />
                        </button>
                        <p className={styles.Oauth2LoginORLabel}>OR</p>
                        <button className={styles.ToggleButton} onClick={this.toggleFormsHandler}>
                            Signup
                        </button>
                    </div>
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

export default Auth;