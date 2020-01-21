import React from 'react';

import styles from './index.module.css';

import GoogleLogo from 'assets/images/GoogleLogo.svg';
import FacebookLogo from 'assets/images/FacebookLogo.svg';

const OAuth2Buttons = props => (
    <div className={styles.OAuth2Panel}>
        <h2>Log in with social accounts</h2>
        <div className={styles.OAuth2Buttons}>
            <button className={`${styles.Button} ${styles.OAuth2Button}`}>
                <img src={GoogleLogo} alt="Google Logo"/>
            </button>
            <button className={`${styles.Button} ${styles.OAuth2Button}`}>
                <img src={FacebookLogo} alt="Facebook Logo"/>
            </button>
        </div>
        <p className={styles.ORLabel}>OR</p>
        <button className={`${styles.Button} ${styles.SwitchButton}`} onClick={props.switch}>
            {props.switchButtonContent}
        </button>
    </div>
);

export default OAuth2Buttons;