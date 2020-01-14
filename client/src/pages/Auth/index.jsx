import React, { Component } from 'react';

import AuthForm from 'components/AuthForm';
import Logo from 'components/Logo';

import styles from './index.module.css';

import EmailImage from 'assets/images/Email.png';
import NameImage from 'assets/images/Name.png';
import PasswordImage from 'assets/images/Password.png';

class Auth extends Component {
    state = {
        loading: false,
        showLogin: 'Login'
    };

    toggleFormsHandler = () => {
        this.setState(prevState => {
            return {
                showLogin: !prevState.showLogin
            };
        })
    };

    authHandler = values => {
        fetch(process.env.REACT_APP_BASE_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(values)
        }).then(data => {
            console.log(data);
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

        return (
            <main className={styles.Auth}>
                <div className={`${styles.FormContainer} ${position}`}>
                    <div className={this.state.showLogin ? styles.Shown : styles.Hide}>
                        <AuthForm inputs={loginInputs} submit={this.authHandler} heading="Log in to your account" buttonContent="LOG IN" />
                    </div>
                    <div className={this.state.showLogin ? styles.Hide : styles.Shown}>
                        <AuthForm inputs={signupInputs} submit={this.authHandler} heading="Create new account" buttonContent="SIGN UP" />
                    </div>
                </div>
                <div className={styles.LoginToggle}>
                    <button onClick={this.toggleFormsHandler}>Log in or continue with 3rd party</button>
                </div>
                <div className={styles.Oauth2Login}>
                    <h2>Login with 3rd party</h2>
                    <button>
                        <Logo src="Google" />
                    </button>
                    <button>
                        <Logo src="Facebook" />
                    </button>
                    <p>OR</p>
                    <button className={styles.SignUpButton} onClick={this.toggleFormsHandler}>
                        Signup
                    </button>
                </div>
            </main>
        );
    }
}

export default Auth;