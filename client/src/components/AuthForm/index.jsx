import React, {Component} from 'react';

import Input from 'components/Input';

import styles from './index.module.css';

class AuthForm extends Component {
    constructor(props) {
        super(props);
        this.state = { inputs: {}, isFormValid: false };
        for (let input in props.inputs) {
            this.state.inputs[input] = {
                ...props.inputs[input]
            }
        }
    }

    changeHandler = (e, inputName) => {
        const [isValid, errorMessage] =
            this.validateInput(this.state.inputs[inputName].validation, e.target.value, inputName);
        let isFormValid = isValid;
        Object.entries(this.state.inputs).forEach(([inputKey, inputValue]) => (
            inputKey !== inputName ? isFormValid = inputValue.isValid && isFormValid : null
        ));
        this.setState({
            ...this.state,
            isFormValid: isFormValid,
            inputs: {
                ...this.state.inputs,
                [inputName]: {
                    ...this.state.inputs[inputName],
                    value: e.target.value,
                    isValid: isValid,
                    isTouched: true,
                    errorMessage: errorMessage
                }
            }
        });
    };

    formSubmitHandler = e => {
        e.preventDefault();
        const values = {};
        for (let input in this.state.inputs) {
            values[input] = this.state.inputs[input].value;
        }
        this.props.submit(values);
    };

    validateInput = (rules, value, name) => {
        let isValid = true;
        let errorMessage = null;
        if (rules.regex && !value.match(rules.regex)) {
            isValid = false;
            errorMessage = 'Invalid ' + name;
        }
        if (rules.maxLength && value.length > rules.maxLength) {
            isValid = false;
            errorMessage = name + ' is too long';
        }
        if (rules.minLength && value.length < rules.minLength) {
            isValid = false;
            errorMessage = name + ' is too short';
        }
        if (rules.isRequired && value === '') {
            isValid = false;
            errorMessage = name + ' is required';
        }
        return [isValid, errorMessage];
    };

    render() {
        return (
            <div className={styles.Login}>
                {this.props.message ? <p style={{color: this.props.messageColor}}>{this.props.message}</p> : null}
                <h1>{this.props.heading}</h1>
                <form onSubmit={this.formSubmitHandler} noValidate>
                    {Object.entries(this.state.inputs).map(([name, input]) => (
                        <Input key={name} {...input} change={e => this.changeHandler(e, name)}/>
                    ))}
                    <button type="submit" className={styles.Button} disabled={!this.state.isFormValid}>
                        {this.props.buttonContent}
                    </button>
                </form>
                <button className={`${styles.Button} ${styles.SwitchButton}`} onClick={this.props.switch}>
                    {this.props.switchContent}
                </button>
            </div>
        )
    }
}

export default AuthForm;