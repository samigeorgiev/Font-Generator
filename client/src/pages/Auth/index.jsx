import React, { Component } from 'react';

import AuthForm from 'components/AuthForm';

import styles from './index.module.css';

import EmailImage from 'assets/images/Email.png';
import NameImage from 'assets/images/Name.png';
import PasswordImage from 'assets/images/Password.png';

class Auth extends Component {
    state = {
        showLogin: 'Login'
    };

    toggleFormsHandler = () => {
        this.setState(prevState => {
            return {
                showLogin: !prevState.showLogin
            };
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
                        <AuthForm inputs={loginInputs} heading="Log in to your account" buttonContent="LOG IN" />
                    </div>
                    <div className={this.state.showLogin ? styles.Hide : styles.Shown}>
                        <AuthForm inputs={signupInputs} heading="Create new account" buttonContent="SIGN UP" />
                    </div>
                </div>
                <button onClick={this.toggleFormsHandler}>
                    Signup
                </button>
            </main>
        );
    }
}

export default Auth;