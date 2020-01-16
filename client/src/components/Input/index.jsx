import React from 'react';

import styles from './index.module.css';

const input = props => {
    const classes = [styles.Input];
    let validationError = null;

    if (!props.isValid && props.isTouched) {
        validationError = (
            <p className={styles.ValidationError}>
                {props.errorMessage}
            </p>
        );
        classes.push(styles.Invalid);
    }

    return (
        <div className={styles.InputContainer}>
            <input
                type={props.type}
                placeholder={props.placeholder}
                className={classes.join(' ')}
                value={props.value}
                onChange={props.change}
                style={{
                    backgroundImage: `url(${props.background})`
                }}
            />
            {validationError}
        </div>
    );
};

export default input;